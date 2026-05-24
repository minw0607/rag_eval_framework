"""
Full multi-prompt evaluation loop with checkpointing, sanity checks,
and low-quality investigation.

Functions
---------
run_evaluation(rag_system, data, embedding_client, generation_client,
               config, output_dir, checkpoint_dir) -> dict
    Run the complete evaluation and return the results dict.
"""

import gc
import json
import os
import time

import numpy as np
from tqdm import tqdm

from src.metrics      import EvalMetrics
from src.cost_tracker import CostTracker
from src.investigator import LowQualityInvestigator


# =============================================================================
# Public API
# =============================================================================

def run_evaluation(
    rag_system,
    data,
    embedding_client,
    generation_client,
    config,
    output_dir: str = "./outputs",
    checkpoint_dir: str = "./checkpoints",
) -> dict:
    """
    Run the full multi-prompt evaluation loop.

    Covers:
    - Multi-prompt generation (all Config.PROMPT_VARIANTS)
    - Quality validation (groundedness, context coverage, faithfulness)
    - Completeness cascade (Tier 1 embedding + Tier 2 LLM judge)
    - Adaptive checkpointing every Config.CHECKPOINT_INTERVAL questions
    - Sanity checks on the loaded results file
    - Low-quality investigation (3-layer: rule-based → cluster → export)

    Parameters
    ----------
    rag_system : EnhancedRAGSystem
        Instantiated RAG system.
    data : list
        Sampled HotpotQA question list.
    embedding_client : openai.OpenAI | openai.AzureOpenAI
        Client used for embedding calls.
    generation_client : openai.OpenAI | openai.AzureOpenAI
        Client used for generation calls.
    config : Config
        Framework configuration.
    output_dir : str
        Directory for results JSON and plots (default: Config.OUTPUT_DIR).
    checkpoint_dir : str
        Directory for checkpoint files (default: Config.CHECKPOINT_DIR).

    Returns
    -------
    dict
        The full results dict that was saved to JSON
        (keys: metadata, aggregated_results, detailed_results).
    """
    vector_store = rag_system.vector_store

    EVAL_SAMPLE_SIZE    = config.NUM_EVAL_QUESTIONS
    CHECKPOINT_INTERVAL = config.CHECKPOINT_INTERVAL

    ENABLE_QUALITY_VALIDATION   = config.QUALITY_VALIDATION_ENABLED
    ENABLE_RAGAS_FAITHFULNESS   = config.QUALITY_VALIDATION_ENABLED
    ENABLE_COMPLETENESS_CASCADE = config.QUALITY_VALIDATION_ENABLED
    ENABLE_COST_TRACKING        = True

    os.makedirs(output_dir,     exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        checkpoint_dir,
        f"multi_prompt_eval_checkpoint_{EVAL_SAMPLE_SIZE}.json"
    )
    results_file = os.path.join(
        output_dir,
        f"multi_prompt_eval_results_{EVAL_SAMPLE_SIZE}.json"
    )

    print("=" * 80)
    print("🚀 MULTI-PROMPT EVALUATION")
    print("=" * 80)
    print(f"  Questions:   {EVAL_SAMPLE_SIZE} × {len(config.PROMPT_VARIANTS)} prompts")
    print(f"  Variants:    {', '.join(config.PROMPT_VARIANTS.keys())}")
    print(f"\n  Quality Validation:      {'✅ ENABLED' if ENABLE_QUALITY_VALIDATION else '❌ Disabled'}")
    if ENABLE_QUALITY_VALIDATION:
        faith_method = 'RAGAS LLM-as-judge' if ENABLE_RAGAS_FAITHFULNESS else 'Embedding MiniMax'
        print(f"  Faithfulness method:     {faith_method}")
    print(f"  Completeness Cascade:    "
          f"{'✅ ENABLED (Tier B + C triggers)' if ENABLE_COMPLETENESS_CASCADE else '❌ Disabled'}")
    print(f"  Cost Tracking:           "
          f"{'✅ ENABLED (generation vs evaluation)' if ENABLE_COST_TRACKING else '❌ Disabled'}")
    print("=" * 80 + "\n")

    # =========================================================================
    # Initialize tracking / resume from checkpoint
    # =========================================================================
    if ENABLE_COST_TRACKING:
        cost_tracker = CostTracker()

    start_idx   = 0
    all_results = []

    if os.path.exists(checkpoint_file) and not config.FORCE_RESTART:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        all_results = checkpoint_data['results']
        start_idx   = len(all_results)
        print(f"📂 Resuming from question {start_idx + 1}/{EVAL_SAMPLE_SIZE}\n")

    eval_questions = data[start_idx:min(start_idx + EVAL_SAMPLE_SIZE, len(data))]

    print(f"🔄 Starting evaluation...")
    print(f"   Questions remaining: {len(eval_questions)}")
    print(f"   Checkpoint every:    {CHECKPOINT_INTERVAL} questions\n")

    start_time = time.time()
    errors     = 0

    # =========================================================================
    # Evaluation loop
    # =========================================================================
    for i, question_item in enumerate(tqdm(eval_questions, desc="Evaluating")):
        try:
            question    = question_item['question']
            gold_answer = question_item['answer']

            # -- Step 1: Retrieve documents -----------------------------------
            retrieved_docs = vector_store.search(
                query=question,
                top_k=config.TOP_K,
                strategy=config.RETRIEVAL_STRATEGY
            )

            # -- Step 2: Retrieval quality ------------------------------------
            hit_rate, _ = EvalMetrics.calculate_hit_rate(question_item, retrieved_docs)

            # -- Step 3: Build context string ---------------------------------
            context_parts = []
            for idx, doc in enumerate(retrieved_docs, 1):
                title   = doc.get('metadata', {}).get('title', f'Document {idx}')
                content = doc.get('content', '')
                context_parts.append(f"Document {idx} ({title}):\n{content[:500]}")
            context = '\n\n'.join(context_parts)

            # -- Step 4: Generate answers for all prompt variants -------------
            variant_results = EvalMetrics.evaluate_multi_prompt(
                question_item=question_item,
                prompt_variants=config.PROMPT_VARIANTS,
                context=context,
                generation_client=generation_client,
                config=config,
                retrieved_docs=retrieved_docs
            )

            if ENABLE_COST_TRACKING:
                ctx_tokens = int(len(context.split()) * 1.3)
                for v_result in variant_results.values():
                    if not v_result.get('is_error', False):
                        ans_tokens = int(v_result['answer_length'] * 1.3)
                        cost_tracker.track_answer_generation(
                            input_tokens=15 + ctx_tokens,
                            output_tokens=ans_tokens,
                            model=config.GENERATION_MODEL
                        )

            # -- Step 5: Quality validation -----------------------------------
            if ENABLE_QUALITY_VALIDATION:
                for variant_name, v_result in variant_results.items():
                    if v_result.get('is_error', False):
                        continue

                    quality_metrics = EvalMetrics.validate_quality(
                        answer=v_result['clean_answer'],
                        context_docs=retrieved_docs,
                        question=question,
                        gold_answer=gold_answer,
                        quality_guard=rag_system.quality_guard if hasattr(rag_system, 'quality_guard') else None,
                        generation_client=generation_client if ENABLE_RAGAS_FAITHFULNESS else None,
                        config=config if ENABLE_RAGAS_FAITHFULNESS else None
                    )
                    v_result.update(quality_metrics)

                    if ENABLE_COST_TRACKING:
                        q_tokens        = int(len(question.split()) * 1.3)
                        ans_tokens      = int(v_result['answer_length'] * 1.3)
                        ctx_tokens_eval = int(len(context.split()) * 1.3)

                        cost_tracker.track_retrieval_embedding(
                            query_tokens=q_tokens,
                            model=vector_store.embedding_model
                        )
                        cost_tracker.track_groundedness_embedding(
                            tokens=ans_tokens + ctx_tokens_eval,
                            model=vector_store.embedding_model
                        )
                        if ENABLE_RAGAS_FAITHFULNESS:
                            cost_tracker.track_faithfulness_llm(
                                input_tokens=int(ctx_tokens_eval + ans_tokens + 200),
                                output_tokens=60,
                                model=config.GENERATION_MODEL
                            )

            # -- Step 6: Completeness cascade ---------------------------------
            if ENABLE_COMPLETENESS_CASCADE and ENABLE_QUALITY_VALIDATION:
                for variant_name, v_result in variant_results.items():
                    if v_result.get('is_error', False):
                        continue

                    cascade_result = EvalMetrics.evaluate_completeness_cascade(
                        answer=v_result['clean_answer'],
                        question=question,
                        context_docs=retrieved_docs,
                        embedding_client=vector_store.embedding_client,
                        generation_client=generation_client,
                        embedding_model=vector_store.embedding_model,
                        generation_model=config.GENERATION_MODEL,
                        groundedness=v_result.get('groundedness'),
                        faithfulness=v_result.get('faithfulness'),
                    )

                    v_result['completeness']          = cascade_result['completeness']
                    v_result['context_coverage']      = cascade_result['context_coverage']
                    v_result['tier2_triggered']       = cascade_result['tier2_triggered']
                    v_result['tier2_trigger_reasons'] = cascade_result['tier2_trigger_reasons']
                    v_result['agreement']             = cascade_result['agreement']

                    if ENABLE_COST_TRACKING:
                        t1_meta    = cascade_result.get('tier1_meta', {})
                        n_texts    = (1 + t1_meta.get('n_context_sentences', 20)
                                     + max(len(v_result['clean_answer'].split()), 1))
                        est_t1_tokens = int(n_texts * 8 * 1.3)
                        cost_tracker.track_completeness_embedding(
                            tokens=est_t1_tokens,
                            model=vector_store.embedding_model
                        )

                        if cascade_result['tier2_triggered']:
                            t2_meta = cascade_result.get('tier2_meta') or {}
                            cost_tracker.track_completeness_llm(
                                input_tokens=t2_meta.get('est_input_tokens', 3300),
                                output_tokens=t2_meta.get('est_output_tokens', 80),
                                model=config.GENERATION_MODEL,
                                triggered=True,
                                tier1_score=cascade_result['tier1_score'],
                                tier2_score=cascade_result['tier2_score'],
                                trigger_reasons=cascade_result['tier2_trigger_reasons'],
                                agreement=cascade_result['agreement']
                            )
                        else:
                            cost_tracker.track_completeness_llm(
                                input_tokens=0, output_tokens=0,
                                model=config.GENERATION_MODEL,
                                triggered=False
                            )

            # -- Step 7: Store result -----------------------------------------
            all_results.append({
                'question':           question,
                'gold_answer':        gold_answer,
                'hit_rate':           hit_rate,
                'num_docs_retrieved': len(retrieved_docs),
                'variant_results':    variant_results
            })

            # -- Step 8: Checkpoint -------------------------------------------
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_data = {
                    'completed': len(all_results),
                    'results':   all_results,
                    'timestamp': str(time.time()),
                }
                if ENABLE_COST_TRACKING:
                    checkpoint_data['cost_summary'] = cost_tracker.get_summary()

                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                elapsed = (time.time() - start_time) / 60
                rate    = len(all_results) / elapsed if elapsed > 0 else 0
                eta     = (EVAL_SAMPLE_SIZE - len(all_results)) / rate if rate > 0 else 0

                print(f"\n✓ {len(all_results)}/{EVAL_SAMPLE_SIZE} | "
                      f"{rate:.1f} q/min | ETA: {eta:.0f}min", end='')
                if ENABLE_COST_TRACKING:
                    print(f" | Gen: ${cost_tracker.get_total_generation_cost():.2f} | "
                          f"Eval: ${cost_tracker.get_total_evaluation_cost():.2f}", end='')
                    if ENABLE_COMPLETENESS_CASCADE:
                        trig_rate = cost_tracker.get_cascade_trigger_rate()
                        if trig_rate is not None:
                            print(f" | Cascade: {trig_rate:.0%}", end='')
                print()
                gc.collect()

        except Exception as e:
            errors += 1
            print(f"\n⚠️  Error on question {i + 1}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Aggregate results
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 AGGREGATING RESULTS")
    print("=" * 80)

    aggregated = {}

    for variant_name in config.PROMPT_VARIANTS.keys():
        variant_data = [
            result['variant_results'][variant_name]
            for result in all_results
            if not result['variant_results'].get(variant_name, {}).get('is_error', False)
        ]
        if not variant_data:
            continue

        acc_contains  = np.mean([v['correct_contains'] for v in variant_data])
        acc_f1        = np.mean([v['correct_f1']       for v in variant_data])
        avg_f1        = np.mean([v['f1_score']         for v in variant_data])
        citation_rate = np.mean([v.get('has_citation', False) for v in variant_data])
        refusal_rate  = np.mean([v.get('is_refusal',   False) for v in variant_data])
        avg_length    = np.mean([v['answer_length']    for v in variant_data])

        def safe_mean(lst, key):
            vals = [v[key] for v in lst if v.get(key) is not None]
            return np.mean(vals) if vals else None

        quality_data = [v for v in variant_data if v.get('groundedness') is not None]

        avg_groundedness     = safe_mean(quality_data, 'groundedness')
        avg_context_coverage = safe_mean(quality_data, 'context_coverage')
        avg_completeness     = safe_mean(quality_data, 'completeness')
        avg_faithfulness     = safe_mean(quality_data, 'faithfulness')
        avg_answer_rel       = safe_mean(quality_data, 'answer_relevancy')
        avg_context_rel      = safe_mean(quality_data, 'context_relevancy')
        avg_quality_score    = safe_mean(quality_data, 'quality_score')
        quality_valid_rate   = (
            np.mean([v.get('quality_valid', False) for v in quality_data])
            if quality_data else None
        )

        cascade_data = [v for v in variant_data if v.get('tier2_triggered') is not None]
        cascade_trigger_rate   = (np.mean([v['tier2_triggered'] for v in cascade_data])
                                   if cascade_data else None)
        cascade_agreement_rate = None
        if cascade_data:
            triggered_cases = [v for v in cascade_data if v['tier2_triggered']]
            if triggered_cases:
                agree_vals = [v['agreement'] for v in triggered_cases
                              if v.get('agreement') is not None]
                cascade_agreement_rate = np.mean(agree_vals) if agree_vals else None

        aggregated[variant_name] = {
            'total_questions':         len(variant_data),
            'accuracy_contains':       acc_contains,
            'accuracy_f1':             acc_f1,
            'avg_f1_score':            avg_f1,
            'citation_rate':           citation_rate,
            'refusal_rate':            refusal_rate,
            'avg_answer_length':       avg_length,
            'avg_groundedness':        avg_groundedness,
            'avg_context_coverage':    avg_context_coverage,
            'avg_completeness':        avg_completeness,
            'avg_faithfulness':        avg_faithfulness,
            'avg_answer_relevancy':    avg_answer_rel,
            'avg_context_relevancy':   avg_context_rel,
            'avg_quality_score':       avg_quality_score,
            'quality_valid_rate':      quality_valid_rate,
            'cascade_trigger_rate':    cascade_trigger_rate,
            'cascade_agreement_rate':  cascade_agreement_rate,
        }

    # =========================================================================
    # Display results table
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80 + "\n")
    print(f"{'Variant':<12} | {'F1≥0.5':<7} | {'Contains':<9} | {'Faith':<6} | "
          f"{'Ground':<7} | {'Complete':<9} | {'Quality':<8} | {'Refusal':<8} | {'Length':<6}")
    print("-" * 100)

    for variant_name, m in aggregated.items():
        print(
            f"{variant_name.capitalize():<12} | "
            f"{m['accuracy_f1']:>6.1%} | "
            f"{m['accuracy_contains']:>8.1%} | "
            f"{(m['avg_faithfulness'] or 0):>5.2f} | "
            f"{(m['avg_groundedness'] or 0):>6.2f} | "
            f"{(m['avg_completeness'] or 0):>8.3f} | "
            f"{(m['avg_quality_score'] or 0):>7.2f} | "
            f"{m['refusal_rate']:>7.1%} | "
            f"{m['avg_answer_length']:>5.1f}w"
        )

    if ENABLE_COMPLETENESS_CASCADE:
        print(f"\n{'─' * 60}")
        print("Completeness Cascade (Tier 2 LLM judge):")
        for variant_name, m in aggregated.items():
            trig  = m.get('cascade_trigger_rate')
            agree = m.get('cascade_agreement_rate')
            trig_str  = f"{trig:.1%}"  if trig  is not None else "N/A"
            agree_str = f"{agree:.1%}" if agree is not None else "N/A"
            print(f"  {variant_name.capitalize():<12} trigger={trig_str:<7}  agreement={agree_str}")

    print(f"\nRetrieval hit rate:  {np.mean([r['hit_rate'] for r in all_results]):.1%}")

    if ENABLE_COST_TRACKING:
        gen_cost  = cost_tracker.get_total_generation_cost()
        eval_cost = cost_tracker.get_total_evaluation_cost()
        n         = max(len(all_results), 1)
        print(f"\nCost breakdown:")
        print(f"  Generation:  ${gen_cost:.4f}  (${gen_cost/n:.4f}/q)")
        print(f"  Evaluation:  ${eval_cost:.4f}  (${eval_cost/n:.4f}/q)")
        print(f"  Total:       ${cost_tracker.get_total_cost():.4f}  "
              f"(${cost_tracker.get_total_cost()/n:.4f}/q)")

    if errors > 0:
        print(f"\n⚠️  Errors: {errors} questions failed")

    # =========================================================================
    # Save results
    # =========================================================================
    cost_summary_meta = None
    if ENABLE_COST_TRACKING:
        cost_summary_meta = {
            'total_cost':             cost_tracker.get_total_cost(),
            'total_generation_cost':  cost_tracker.get_total_generation_cost(),
            'total_evaluation_cost':  cost_tracker.get_total_evaluation_cost(),
            'generation_costs':       cost_tracker.generation_costs,
            'evaluation_costs':       cost_tracker.evaluation_costs,
            'cascade_trigger_rate':   cost_tracker.get_cascade_trigger_rate(),
            'cascade_agreement_rate': cost_tracker.get_cascade_agreement_rate(),
            'cascade_stats':          cost_tracker.cascade_stats,
        }

    final_data = {
        'metadata': {
            'timestamp':             str(time.time()),
            'num_questions':         len(all_results),
            'eval_sample_size':      EVAL_SAMPLE_SIZE,
            'quality_validation':    ENABLE_QUALITY_VALIDATION,
            'ragas_faithfulness':    ENABLE_RAGAS_FAITHFULNESS,
            'completeness_cascade':  ENABLE_COMPLETENESS_CASCADE,
            'faithfulness_method':   ('RAGAS LLM-as-judge'
                                      if ENABLE_RAGAS_FAITHFULNESS else 'Embedding MiniMax'),
            'completeness_method':   ('Cascade (Tier 1 embedding + Tier 2 LLM)'
                                      if ENABLE_COMPLETENESS_CASCADE else 'Embedding only'),
            'variants':              list(config.PROMPT_VARIANTS.keys()),
            'top_k':                 config.TOP_K,
            'retrieval_strategy':    config.RETRIEVAL_STRATEGY,
            'generation_model':      config.GENERATION_MODEL,
            'embedding_model':       vector_store.embedding_model,
            'errors':                errors,
            'total_cost':            (cost_tracker.get_total_cost()
                                      if ENABLE_COST_TRACKING else None),
            'cost_summary':          cost_summary_meta,
        },
        'aggregated_results': aggregated,
        'detailed_results':   all_results,
    }

    with open(results_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n✓ Checkpoint cleaned up")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  Questions evaluated: {len(all_results)}/{EVAL_SAMPLE_SIZE}")
    print(f"  Time elapsed:        {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Errors:              {errors}")
    if ENABLE_COST_TRACKING:
        print(f"  Generation cost:     ${cost_tracker.get_total_generation_cost():.4f}")
        print(f"  Evaluation cost:     ${cost_tracker.get_total_evaluation_cost():.4f}")
        print(f"  Total cost:          ${cost_tracker.get_total_cost():.4f}")
    print(f"  Results saved:       {results_file}")
    print("=" * 80 + "\n")

    # =========================================================================
    # Sanity checks on saved results
    # =========================================================================
    _run_sanity_checks(results_file)

    # =========================================================================
    # Low-quality investigation
    # =========================================================================
    _run_investigation(results_file, output_dir)

    return final_data


# =============================================================================
# Internal helpers
# =============================================================================

def _run_sanity_checks(results_file: str) -> None:
    """Load results and print diagnostic checks (mirrors notebook cell 28)."""
    import glob

    print("=" * 80)
    print("🔍 EVALUATION RESULTS DIAGNOSTIC")
    print("=" * 80)

    try:
        with open(results_file, 'r') as f:
            eval_data = json.load(f)

        print(f"\n✅ Loaded: {results_file}")

        if 'metadata' in eval_data:
            print(f"\n📋 Metadata:")
            for key, value in eval_data['metadata'].items():
                if key not in ('cost_summary',):   # skip nested dicts
                    print(f"   {key}: {value}")

        if 'detailed_results' in eval_data and len(eval_data['detailed_results']) > 0:
            diag_first   = eval_data['detailed_results'][0]
            variants     = list(diag_first['variant_results'].keys())
            all_results  = eval_data['detailed_results']
            total        = len(all_results)
            print(f"\n✅ {total} questions | {len(variants)} variants: {', '.join(variants)}")
        else:
            print("\n⚠️  No detailed results found — skipping sanity checks")
            return

        # Sample individual results
        print(f"\n{'=' * 80}")
        print("CHECK: Sample Individual Results (First 3 Questions)")
        print("=" * 80)
        for i, result in enumerate(all_results[:3]):
            print(f"\n📝 Question {i+1}: {result['question'][:80]}...")
            print(f"   Gold: {result['gold_answer']}")
            for variant in variants:
                vr = result['variant_results'][variant]
                print(f"   {variant.upper()}:")
                print(f"      Answer:  {vr.get('clean_answer', 'N/A')[:60]}...")
                print(f"      F1:      {vr.get('f1_score', 0):.2f}")
                faith = vr.get('faithfulness')
                if faith is not None:
                    print(f"      Faith:   {faith:.2f}")
                else:
                    print("      Faith:   None ⚠️")

        # Metric coverage
        print(f"\n{'=' * 80}")
        print("CHECK: Metric Availability")
        print("=" * 80)
        metrics_to_check = [
            'f1_score', 'groundedness', 'completeness', 'faithfulness',
            'answer_relevancy', 'context_relevancy', 'quality_score'
        ]
        for variant in variants:
            print(f"\n📊 {variant.upper()}:")
            for metric in metrics_to_check:
                available = sum(
                    1 for r in all_results
                    if r['variant_results'][variant].get(metric) is not None
                )
                pct = available / total * 100
                status = "✅" if pct >= 99 else "⚠️"
                print(f"   {metric:<22} {available}/{total} ({pct:.0f}%) {status}")

        print("\n✅ DIAGNOSTIC COMPLETE\n")

    except Exception as e:
        print(f"⚠️  Sanity check failed: {e}")


def _run_investigation(results_file: str, output_dir: str) -> None:
    """Run the 3-layer low-quality investigation (mirrors notebook cell 30)."""
    print("=" * 80)
    print("🔍 QUALITY INVESTIGATOR — 3-Layer Hybrid")
    print("=" * 80)

    try:
        investigator = LowQualityInvestigator(results_file=results_file)

        inv_flagged = investigator.find_low_quality_answers(
            f1_threshold=0.3,
            groundedness_threshold=0.5,
            faithfulness_threshold=0.5,
            completeness_threshold=0.4,
            quality_score_threshold=0.55,
        )
        print(f"Found {len(inv_flagged)} flagged answers")

        investigator.print_pattern_summary(inv_flagged)
        investigator.print_worst_cases(inv_flagged, num_cases=10)

        # Layer 2: cluster analysis
        inv_cluster_result = investigator.cluster_unk_answers(inv_flagged)
        if inv_cluster_result:
            investigator.print_cluster_summary(inv_cluster_result)

        # Layer 3: export
        lq_dir = os.path.join(output_dir, 'low_quality_analysis')
        inv_export = investigator.export_low_quality_report(
            inv_flagged,
            output_dir=lq_dir,
            top_worst_cases=20
        )
        if inv_cluster_result:
            investigator.export_cluster_cases(inv_cluster_result, output_dir=lq_dir)

        print(f"\n📁 Outputs written to: {lq_dir}/")
        print(f"   low_quality_summary.csv      ← all flagged answers")
        print(f"   pattern_summary.json         ← pattern frequency + mitigations")
        print(f"   worst_cases/*.md             ← top 20 individual case reports")
        if inv_cluster_result:
            print(f"   unk_clusters/*.md            ← {inv_cluster_result['k']} cluster review files")

    except Exception as e:
        print(f"⚠️  Investigation skipped: {e}")
