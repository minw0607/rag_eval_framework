"""
Pre-evaluation system check for the RAG evaluation framework.

Validates all 13 framework components before the full evaluation run.
Fix any ❌ before proceeding to the evaluation loop.
"""

import traceback

import numpy as np

from src.metrics      import EvalMetrics
from src.cost_tracker import CostTracker


# =============================================================================
# Public API
# =============================================================================

def run_preflight_check(data, vector_store, embedding_client, generation_client, config) -> bool:
    """
    Run all 13 pre-evaluation checks and print results.

    Parameters
    ----------
    data : list
        Sampled HotpotQA question list.
    vector_store : VectorStore
        Built and client-injected vector store.
    embedding_client : openai.OpenAI | openai.AzureOpenAI
        Client used for embedding calls.
    generation_client : openai.OpenAI | openai.AzureOpenAI
        Client used for generation calls.
    config : Config
        Framework configuration.

    Returns
    -------
    bool
        True if all checks pass, False if any check failed.
    """
    from src.quality_guard import QualityGuard

    print("=" * 80)
    print("🔍 COMPREHENSIVE PRE-EVALUATION SYSTEM CHECK")
    print("=" * 80)
    print("\nChecks: Data | Vector Store | RAG System | Quality Guard |")
    print("        Prompts | EvalMetrics | Cost Tracker | NEW: Completeness Cascade")
    print("=" * 80)

    chk_passed = []
    chk_failed = []

    def chk_ok(label):
        chk_passed.append(label)

    def chk_fail(label):
        chk_failed.append(label)

    # =========================================================================
    # CHECK 1: Data
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 1: Data")
    print("─" * 60)
    try:
        assert len(data) > 0, "data is empty"
        chk_q = data[0]
        assert 'question' in chk_q, "missing 'question' key"
        assert 'answer'   in chk_q, "missing 'answer' key"
        assert 'context'  in chk_q, "missing 'context' key"
        print(f"✅ {len(data):,} questions loaded")
        print(f"   Sample Q:    {chk_q['question'][:80]}...")
        print(f"   Sample A:    {chk_q['answer']}")
        print(f"   Gold docs:   {', '.join([t for t, _ in chk_q['context'][:2]])}")
        chk_ok("Data")
    except Exception as e:
        print(f"❌ Data: {e}")
        chk_fail("Data")

    # =========================================================================
    # CHECK 2: Vector Store
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 2: Vector Store")
    print("─" * 60)
    try:
        assert vector_store is not None
        assert len(vector_store.documents) > 0
        assert vector_store.embeddings is not None
        assert vector_store.embedding_client is not None, \
            "embedding_client not re-injected after cache load — call set_clients()"

        chk_emb_resp = vector_store.embedding_client.embeddings.create(
            input=["smoke test"], model=vector_store.embedding_model
        )
        assert len(chk_emb_resp.data[0].embedding) > 0

        print(f"✅ Vector store ready")
        print(f"   Documents:   {len(vector_store.documents):,}")
        print(f"   Embeddings:  {vector_store.embeddings.shape}")
        print(f"   Model:       {vector_store.embedding_model}")
        print(f"   Strategy:    {config.RETRIEVAL_STRATEGY}")
        print(f"   Client:      ✓ (embedding smoke test passed)")
        chk_ok("Vector Store")
    except Exception as e:
        print(f"❌ Vector Store: {e}")
        chk_fail("Vector Store")

    # =========================================================================
    # CHECK 3: Generation Client
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 3: Generation Client")
    print("─" * 60)
    try:
        chk_gen_resp = generation_client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": "Reply with: OK"}],
            max_tokens=5
        )
        chk_gen_out = chk_gen_resp.choices[0].message.content.strip()
        print(f"✅ Generation client ready")
        print(f"   Model:       {config.GENERATION_MODEL}")
        print(f"   Response:    '{chk_gen_out}'")
        print(f"   Max tokens:  {config.MAX_TOKENS}")
        chk_ok("Generation Client")
    except Exception as e:
        print(f"❌ Generation Client: {e}")
        chk_fail("Generation Client")

    # =========================================================================
    # CHECK 4: Quality Guard
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 4: Quality Guard")
    print("─" * 60)
    try:
        quality_guard = QualityGuard(
            embedding_client=embedding_client,
            embedding_model=config.EMBEDDING_MODEL,
            config=config,
        )
        assert quality_guard is not None
        chk_qg_info = quality_guard.get_info()
        print(f"✅ Quality Guard ready")
        print(f"   Model:       {chk_qg_info['embedding_model']}")
        print(f"   Provider:    {chk_qg_info['provider']}")
        cost_str = "FREE (local)" if chk_qg_info['provider'].startswith('local') else "API (see pricing table)"
        print(f"   Cost:        {cost_str}")
        print(f"   Thresholds:  groundedness≥{chk_qg_info['thresholds'].get('groundedness', 0.5):.2f} | "
              f"completeness≥{chk_qg_info['thresholds'].get('completeness', 0.4):.2f}")
        chk_ok("Quality Guard")
    except Exception as e:
        print(f"⚠️  Quality Guard not available (optional): {e}")
        print(f"   Evaluation will continue without quality validation")

    # =========================================================================
    # CHECK 5: EvalMetrics — Core Methods
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 5: EvalMetrics — Core Methods")
    print("─" * 60)
    try:
        chk_f1, chk_correct = EvalMetrics.calculate_f1("John Doman", "John Doman")
        assert chk_f1 == 1.0 and chk_correct, f"F1 mismatch: {chk_f1}"

        assert EvalMetrics.contains_match("John Doman is correct", "John Doman") == 1.0
        assert EvalMetrics.exact_match("the cat", "cat") == 1.0
        assert EvalMetrics.detect_refusal("Information not available") is True
        assert EvalMetrics.detect_refusal("John Doman") is False

        chk_parsed = EvalMetrics.parse_structured_response("Answer: John Doman\nSources: Doc 1")
        assert chk_parsed['answer'] == "John Doman"
        assert chk_parsed['parsed_successfully'] is True

        print(f"✅ EvalMetrics core methods verified")
        print(f"   calculate_f1:              ✓  (1.0 on exact match)")
        print(f"   contains_match:            ✓")
        print(f"   exact_match:               ✓  (article stripping works)")
        print(f"   detect_refusal:            ✓  (True/False correctly)")
        print(f"   parse_structured_response: ✓")
        chk_ok("EvalMetrics Core")
    except Exception as e:
        print(f"❌ EvalMetrics Core: {e}")
        chk_fail("EvalMetrics Core")

    # =========================================================================
    # CHECK 6: EvalMetrics — Cascade Methods
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 6: EvalMetrics — Completeness Cascade Methods (NEW)")
    print("─" * 60)
    try:
        assert hasattr(EvalMetrics, '_split_sentences'),                "missing _split_sentences"
        assert hasattr(EvalMetrics, 'calculate_completeness_tier1'),   "missing calculate_completeness_tier1"
        assert hasattr(EvalMetrics, '_should_trigger_completeness_llm'), "missing _should_trigger_completeness_llm"
        assert hasattr(EvalMetrics, 'calculate_completeness_tier2'),   "missing calculate_completeness_tier2"
        assert hasattr(EvalMetrics, 'evaluate_completeness_cascade'),  "missing evaluate_completeness_cascade"
        print(f"✅ All cascade methods present")

        chk_sents = EvalMetrics._split_sentences(
            "John Doman is an actor. He appeared in The Wire. He also starred in Emmett's Mark."
        )
        assert len(chk_sents) == 3, f"Expected 3 sentences, got {len(chk_sents)}"
        print(f"   _split_sentences:          ✓  ({len(chk_sents)} sentences extracted)")

        chk_trigger, chk_reasons = EvalMetrics._should_trigger_completeness_llm(
            completeness_t1=0.3, groundedness=0.85, faithfulness=0.9,
            answer="John Doman", n_relevant_sentences=2
        )
        assert chk_trigger is True, "Tier B should have triggered"
        assert any("Tier B" in r for r in chk_reasons), "Tier B reason missing"
        print(f"   _should_trigger (Tier B):  ✓  ({len(chk_reasons)} reason(s))")

        chk_trigger_c, chk_reasons_c = EvalMetrics._should_trigger_completeness_llm(
            completeness_t1=0.6, groundedness=0.65, faithfulness=0.7,
            answer="Yes", n_relevant_sentences=4
        )
        assert chk_trigger_c is True, "Tier C should have triggered"
        assert any("Tier C" in r for r in chk_reasons_c), "Tier C reason missing"
        print(f"   _should_trigger (Tier C):  ✓  (short answer + rich context)")

        chk_no_trigger, _ = EvalMetrics._should_trigger_completeness_llm(
            completeness_t1=0.8, groundedness=0.82, faithfulness=0.85,
            answer="Sugar Ray was formed first in 1986", n_relevant_sentences=2
        )
        assert chk_no_trigger is False, "Should NOT have triggered"
        print(f"   _should_trigger (no trigger): ✓  (clear pass correctly skipped)")

        chk_ok("EvalMetrics Cascade")
    except Exception as e:
        print(f"❌ EvalMetrics Cascade: {e}")
        traceback.print_exc()
        chk_fail("EvalMetrics Cascade")

    # =========================================================================
    # CHECK 7: Completeness Tier 1 — Live Embedding Test
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 7: Completeness Tier 1 — Live Embedding Test")
    print("─" * 60)
    try:
        chk_t1_docs = [{'content': (
            "Sugar Ray is an American rock band formed in Newport Beach, California in 1986. "
            "American Standards is a post-hardcore band formed in Phoenix, Arizona in 2008. "
            "The two bands have very different musical styles."
        )}]
        chk_t1_question = "Which band was formed first, Sugar Ray or American Standards?"

        chk_t1_short, chk_t1_meta_short = EvalMetrics.calculate_completeness_tier1(
            answer="Sugar Ray",
            question=chk_t1_question,
            context_docs=chk_t1_docs,
            embedding_client=vector_store.embedding_client,
            model=vector_store.embedding_model,
        )
        chk_t1_long, _ = EvalMetrics.calculate_completeness_tier1(
            answer="Sugar Ray was formed first in 1986, while American Standards formed in 2008.",
            question=chk_t1_question,
            context_docs=chk_t1_docs,
            embedding_client=vector_store.embedding_client,
            model=vector_store.embedding_model,
        )

        assert chk_t1_short is not None, "Tier 1 returned None for short answer"
        assert chk_t1_long  is not None, "Tier 1 returned None for long answer"

        print(f"✅ Completeness Tier 1 live test passed")
        print(f"   Short answer score:    {chk_t1_short:.3f}  "
              f"(relevant sentences: {chk_t1_meta_short['n_relevant_sentences']})")
        print(f"   Detailed answer score: {chk_t1_long:.3f}")
        print(f"   Context sentences:     {chk_t1_meta_short['n_context_sentences']}")
        print(f"   Relevance threshold:   {chk_t1_meta_short['relevance_threshold']}")

        chk_t1_refusal, _ = EvalMetrics.calculate_completeness_tier1(
            answer="Information not available",
            question=chk_t1_question,
            context_docs=chk_t1_docs,
            embedding_client=vector_store.embedding_client,
            model=vector_store.embedding_model,
        )
        assert chk_t1_refusal == 0.0, f"Refusal should score 0.0, got {chk_t1_refusal}"
        print(f"   Refusal score:         {chk_t1_refusal:.3f}  ✓ (correctly 0.0)")
        chk_ok("Completeness Tier 1")
    except Exception as e:
        print(f"❌ Completeness Tier 1: {e}")
        traceback.print_exc()
        chk_fail("Completeness Tier 1")

    # =========================================================================
    # CHECK 8: Completeness Tier 2 — LLM Judge Smoke Test
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 8: Completeness Tier 2 — LLM Judge Smoke Test")
    print("─" * 60)
    try:
        chk_t2_docs = [{'content': (
            "Sugar Ray is an American rock band formed in 1986. "
            "American Standards is a post-hardcore band formed in 2008."
        )}]
        chk_t2_score, chk_t2_meta = EvalMetrics.calculate_completeness_tier2(
            answer="Sugar Ray",
            question="Which band was formed first?",
            context_docs=chk_t2_docs,
            generation_client=generation_client,
            model=config.GENERATION_MODEL
        )
        assert chk_t2_score is not None, "Tier 2 returned None"
        assert 0.0 <= chk_t2_score <= 1.0, f"Score out of range: {chk_t2_score}"
        assert 'latency_ms' in chk_t2_meta
        assert 'llm_reason' in chk_t2_meta

        print(f"✅ Completeness Tier 2 LLM judge smoke test passed")
        print(f"   Score:      {chk_t2_score:.3f}")
        print(f"   Reason:     {chk_t2_meta['llm_reason'][:80]}")
        print(f"   Latency:    {chk_t2_meta['latency_ms']} ms")
        print(f"   Est tokens: {chk_t2_meta['est_input_tokens']} in / "
              f"{chk_t2_meta['est_output_tokens']} out")
        chk_ok("Completeness Tier 2")
    except Exception as e:
        print(f"❌ Completeness Tier 2: {e}")
        traceback.print_exc()
        chk_fail("Completeness Tier 2")

    # =========================================================================
    # CHECK 9: Full Cascade — End-to-End
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 9: Completeness Cascade — Full End-to-End")
    print("─" * 60)
    try:
        chk_cas_docs = [{'content': (
            "Sugar Ray is an American rock band formed in Newport Beach, California in 1986. "
            "They are best known for their hit songs Fly and Every Morning. "
            "American Standards is a post-hardcore band formed in Phoenix, Arizona in 2008."
        )}]
        chk_cas_result = EvalMetrics.evaluate_completeness_cascade(
            answer="Sugar Ray",
            question="Which band was formed first, Sugar Ray or American Standards?",
            context_docs=chk_cas_docs,
            embedding_client=vector_store.embedding_client,
            generation_client=generation_client,
            embedding_model=vector_store.embedding_model,
            generation_model=config.GENERATION_MODEL,
            groundedness=0.85,
            faithfulness=0.90,
        )

        required_keys = [
            'completeness', 'context_coverage', 'tier1_score',
            'tier2_score', 'tier2_triggered', 'tier2_trigger_reasons',
            'tier1_meta', 'tier2_meta', 'agreement'
        ]
        for key in required_keys:
            assert key in chk_cas_result, f"Missing key: {key}"

        print(f"✅ Full cascade end-to-end test passed")
        print(f"   Context Coverage (T1): {chk_cas_result['context_coverage']:.3f}")
        print(f"   Completeness (final):  {chk_cas_result['completeness']:.3f}")
        print(f"   Tier 2 triggered:      {chk_cas_result['tier2_triggered']}")
        if chk_cas_result['tier2_triggered']:
            print(f"   Tier 2 score:          {chk_cas_result['tier2_score']:.3f}")
            print(f"   Agreement:             {chk_cas_result['agreement']}")
            for reason in chk_cas_result['tier2_trigger_reasons']:
                print(f"   Trigger reason:        {reason}")
        print(f"   Relevant sentences:    "
              f"{chk_cas_result['tier1_meta'].get('n_relevant_sentences', 'N/A')}")
        chk_ok("Completeness Cascade")
    except Exception as e:
        print(f"❌ Completeness Cascade: {e}")
        traceback.print_exc()
        chk_fail("Completeness Cascade")

    # =========================================================================
    # CHECK 10: Cost Tracker
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 10: Cost Tracker — Separated Generation vs Evaluation")
    print("─" * 60)
    try:
        chk_tracker = CostTracker()
        chk_tracker.track_answer_generation(
            input_tokens=3000, output_tokens=50, model=config.GENERATION_MODEL
        )
        chk_tracker.track_retrieval_embedding(query_tokens=15, model=vector_store.embedding_model)
        chk_tracker.track_completeness_embedding(tokens=500, model=vector_store.embedding_model)
        chk_tracker.track_faithfulness_llm(
            input_tokens=500, output_tokens=50, model=config.GENERATION_MODEL
        )
        chk_tracker.track_completeness_llm(
            input_tokens=3300, output_tokens=80, model=config.GENERATION_MODEL,
            triggered=True, tier1_score=0.4, tier2_score=0.7,
            trigger_reasons=["Tier B: test"], agreement=False
        )

        chk_gen_cost   = chk_tracker.get_total_generation_cost()
        chk_eval_cost  = chk_tracker.get_total_evaluation_cost()
        chk_trig_rate  = chk_tracker.get_cascade_trigger_rate()
        chk_agree_rate = chk_tracker.get_cascade_agreement_rate()

        assert chk_gen_cost  > 0, "Generation cost should be > 0"
        assert chk_eval_cost > 0, "Evaluation cost should be > 0"
        assert chk_trig_rate  == 1.0, f"Trigger rate should be 1.0, got {chk_trig_rate}"
        assert chk_agree_rate == 0.0, f"Agreement rate should be 0.0, got {chk_agree_rate}"

        # Backward-compatible aliases
        chk_tracker2 = CostTracker()
        chk_tracker2.track_embedding(input_tokens=15)
        chk_tracker2.track_llm_generation(input_tokens=100, output_tokens=20)
        assert chk_tracker2.get_total_cost() > 0, "Aliases not working"

        print(f"✅ Cost Tracker verified (generation vs evaluation separated)")
        print(f"   Generation cost:   ${chk_gen_cost:.6f}")
        print(f"   Evaluation cost:   ${chk_eval_cost:.6f}")
        print(f"   Eval overhead:     "
              f"{chk_eval_cost / (chk_gen_cost + chk_eval_cost):.1%} of total")
        print(f"   Cascade trigger:   {chk_trig_rate:.0%}  ✓")
        print(f"   Cascade agreement: {chk_agree_rate:.0%}  ✓")
        print(f"   Backward aliases:  ✓")
        chk_ok("Cost Tracker")
    except Exception as e:
        print(f"❌ Cost Tracker: {e}")
        traceback.print_exc()
        chk_fail("Cost Tracker")

    # =========================================================================
    # CHECK 11: Prompt Variants
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 11: Prompt Variants")
    print("─" * 60)
    try:
        assert len(config.PROMPT_VARIANTS) > 0, "No prompt variants defined"
        chk_prompt_issues = []

        for pname, ptemplate in config.PROMPT_VARIANTS.items():
            if '{context}'  not in ptemplate:
                chk_prompt_issues.append(f"{pname}: missing {{context}}")
            if '{question}' not in ptemplate:
                chk_prompt_issues.append(f"{pname}: missing {{question}}")

        if chk_prompt_issues:
            for issue in chk_prompt_issues:
                print(f"   ⚠️  {issue}")
            chk_fail("Prompt Variants")
        else:
            for pname, ptemplate in config.PROMPT_VARIANTS.items():
                print(f"   ✓ {pname:<12} ({len(ptemplate)} chars)")
            print(f"✅ {len(config.PROMPT_VARIANTS)} prompt variants valid")
            chk_ok("Prompt Variants")
    except Exception as e:
        print(f"❌ Prompt Variants: {e}")
        chk_fail("Prompt Variants")

    # =========================================================================
    # CHECK 12: Cost Estimate
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 12: Cost Estimate for Planned Evaluation Run")
    print("─" * 60)
    try:
        chk_est_tracker = CostTracker()
        chk_est = chk_est_tracker.estimate_evaluation_cost(
            num_questions=config.NUM_EVAL_QUESTIONS,
            num_prompts=len(config.PROMPT_VARIANTS),
            generation_model=config.GENERATION_MODEL,
            embedding_model=vector_store.embedding_model,
            cascade_trigger_rate=0.20
        )
        print(f"✅ Cost estimate for planned run:")
        print(f"   Questions:            {config.NUM_EVAL_QUESTIONS:,}")
        print(f"   Prompt variants:      {len(config.PROMPT_VARIANTS)}")
        print(f"   Generation cost:      ${chk_est['total_generation_cost']:.2f}")
        print(f"   Evaluation cost:      ${chk_est['total_evaluation_cost']:.2f}")
        print(f"   Total estimated:      ${chk_est['total_cost']:.2f}")
        print(f"   Per question:         ${chk_est['per_query_total']:.4f}")
        print(f"   (Cascade at ~20% trigger rate)")
        chk_ok("Cost Estimate")
    except Exception as e:
        print(f"⚠️  Cost estimate unavailable: {e}")

    # =========================================================================
    # CHECK 13: Full Pipeline — Single Question End-to-End
    # =========================================================================
    print("\n" + "─" * 60)
    print("CHECK 13: Full Pipeline — Single Question (All Metrics)")
    print("─" * 60)
    try:
        chk_pipe_item     = data[0]
        chk_pipe_question = chk_pipe_item['question']
        chk_pipe_gold     = chk_pipe_item['answer']

        print(f"   Question:  {chk_pipe_question[:70]}...")
        print(f"   Gold:      {chk_pipe_gold}")
        print(f"   Running pipeline...")

        chk_pipe_docs = vector_store.search(chk_pipe_question)
        chk_hit, _    = EvalMetrics.calculate_hit_rate(chk_pipe_item, chk_pipe_docs)

        chk_pipe_ctx = "\n\n".join([
            f"Doc {i+1} ({d.get('metadata', {}).get('title', '')[:40]}): "
            f"{d.get('content', '')[:200]}..."
            for i, d in enumerate(chk_pipe_docs[:5])
        ])

        chk_pipe_prompt = list(config.PROMPT_VARIANTS.values())[0]
        chk_pipe_prompt = chk_pipe_prompt.replace('{context}', chk_pipe_ctx)
        chk_pipe_prompt = chk_pipe_prompt.replace('{question}', chk_pipe_question)

        chk_pipe_resp = generation_client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": chk_pipe_prompt}],
            max_tokens=config.MAX_TOKENS
        )
        chk_pipe_raw    = chk_pipe_resp.choices[0].message.content.strip()
        chk_pipe_parsed = EvalMetrics.parse_structured_response(chk_pipe_raw)
        chk_pipe_ans    = chk_pipe_parsed['answer']

        chk_pipe_f1, chk_pipe_correct = EvalMetrics.calculate_f1(chk_pipe_ans, chk_pipe_gold)
        chk_pipe_contains = EvalMetrics.contains_match(chk_pipe_ans, chk_pipe_gold)
        chk_pipe_refusal  = EvalMetrics.detect_refusal(chk_pipe_ans)

        quality_guard = QualityGuard(
            embedding_client=embedding_client,
            embedding_model=config.EMBEDDING_MODEL,
            config=config,
        )
        chk_pipe_quality = EvalMetrics.validate_quality(
            answer=chk_pipe_ans,
            context_docs=chk_pipe_docs,
            question=chk_pipe_question,
            gold_answer=chk_pipe_gold,
            quality_guard=quality_guard,
            generation_client=generation_client,
            config=config
        )

        chk_pipe_cascade = EvalMetrics.evaluate_completeness_cascade(
            answer=chk_pipe_ans,
            question=chk_pipe_question,
            context_docs=chk_pipe_docs,
            embedding_client=vector_store.embedding_client,
            generation_client=generation_client,
            embedding_model=vector_store.embedding_model,
            generation_model=config.GENERATION_MODEL,
            groundedness=chk_pipe_quality.get('groundedness'),
            faithfulness=chk_pipe_quality.get('faithfulness'),
        )

        print(f"\n✅ Full pipeline passed — results:")
        print(f"   Answer:           '{chk_pipe_ans}'")
        print(f"   F1:               {chk_pipe_f1:.3f}  ({'✓' if chk_pipe_correct else '✗'})")
        print(f"   Contains:         {chk_pipe_contains:.0f}")
        print(f"   Refusal:          {chk_pipe_refusal}")
        print(f"   Hit rate:         {chk_hit:.1%}")
        print(f"   Groundedness:     {chk_pipe_quality.get('groundedness', 'N/A')}")
        print(f"   Context Coverage: {chk_pipe_cascade['context_coverage']:.3f}  (Tier 1)")
        print(f"   Completeness:     {chk_pipe_cascade['completeness']:.3f}  "
              f"({'Tier 2' if chk_pipe_cascade['tier2_triggered'] else 'Tier 1'})")
        print(f"   Faithfulness:     {chk_pipe_quality.get('faithfulness', 'N/A')}")
        print(f"   Quality Score:    {chk_pipe_quality.get('quality_score', 'N/A')}")
        print(f"   T2 triggered:     {chk_pipe_cascade['tier2_triggered']}")
        if chk_pipe_cascade['tier2_triggered']:
            for reason in chk_pipe_cascade['tier2_trigger_reasons']:
                print(f"     → {reason}")
        chk_ok("Full Pipeline")
    except Exception as e:
        print(f"❌ Full Pipeline: {e}")
        traceback.print_exc()
        chk_fail("Full Pipeline")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("📋 PRE-EVALUATION CHECK SUMMARY")
    print("=" * 80)

    print(f"\n  ✅ Passed ({len(chk_passed)}): {', '.join(chk_passed)}")
    if chk_failed:
        print(f"  ❌ Failed ({len(chk_failed)}): {', '.join(chk_failed)}")
        print("\n" + "=" * 80)
        print("⚠️  FIX ALL FAILURES BEFORE RUNNING THE EVALUATION")
        print("=" * 80)
        return False

    print(f"\n  🎯 All checks passed — framework ready for full evaluation")
    print(f"\n  📋 Run configuration:")
    print(f"     Questions:        {config.NUM_EVAL_QUESTIONS:,}")
    print(f"     Prompt variants:  {len(config.PROMPT_VARIANTS)}")
    print(f"     RAGAS faith:      {'✅' if config.QUALITY_VALIDATION_ENABLED else '❌'}")
    print(f"     Quality valid:    {'✅' if config.QUALITY_VALIDATION_ENABLED else '❌'}")
    print(f"     Completeness cas: ENABLED (Tier B + C triggers)")
    print("=" * 80)
    print("✅ ✅ ✅  READY — proceed to Run Evaluation")
    print("=" * 80)
    return True
