"""
Low Quality Answer Investigator — 3-Layer Hybrid Approach

Layer 1 — 16 rule-based patterns classify failure modes
Layer 2 — k-means clustering groups unclassified (UNK) cases  
Layer 3 — Markdown export for human review

Usage
-----
from src.investigator import LowQualityInvestigator

investigator = LowQualityInvestigator(results_file="results/eval_results.json")
flagged  = investigator.find_low_quality_answers(f1_threshold=0.3)
investigator.print_pattern_summary(flagged)
investigator.print_worst_cases(flagged, num_cases=10)

# Cluster unclassified (UNK) cases
clusters = investigator.cluster_unk_answers(flagged)
investigator.print_cluster_summary(clusters)
investigator.export_cluster_cases(clusters, output_dir="results/low_quality_analysis")
"""

import json
import os
import glob
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class LowQualityInvestigator:
    """
    Investigate and categorize low-quality RAG answers for audit and human review.

    3-Layer Hybrid Workflow:
      Layer 1 — Rule-Based (all flagged answers):
        find_low_quality_answers() — flags + assigns pattern P1-P15 or UNK
        print_pattern_summary()   — frequency table + mitigations
        print_worst_cases()       — console display

      Layer 2 — Cluster Analysis (UNK cases only):
        cluster_unk_answers()     — K-Means on metric vectors, auto-selects K
        print_cluster_summary()   — per-cluster metric profiles
        Exports cluster centroids as markdown for human review

      Layer 3 — Human Review:
        export_low_quality_report() — CSV + JSON + per-case markdown
        Cluster representatives flagged for promotion to new rules
    """

    # =========================================================================
    # DIAGNOSTIC PATTERN DEFINITIONS
    # =========================================================================

    PATTERN_DEFINITIONS = {
        'P1':  {
            'name':  'Hallucinated Details',
            'desc':  'Answer is correct (high F1) but contains hallucinated supporting info',
            'mitigations': [
                "Strengthen grounding: 'Only use info from provided documents'",
                "Add explicit refusal instruction for uncertain claims",
                "Switch to Citation prompt variant to force source attribution",
                "Post-validate: reject answers with ungrounded supporting claims",
            ]
        },
        'P2':  {
            'name':  'Verbose / Off-Target Answer',
            'desc':  'Answer is grounded and faithful but too verbose or misses the key answer',
            'mitigations': [
                "Add conciseness constraint: 'Answer in 1-5 words'",
                "Add few-shot examples of concise correct answers",
                "Use Baseline or Concise prompt variant",
                "Post-process: extract key entity from verbose answer",
            ]
        },
        'P3':  {
            'name':  'Tangential Answer',
            'desc':  'Longer answer contains correct info but does not directly address the question',
            'mitigations': [
                "Focus prompt on question: 'Answer the specific question asked'",
                "Repeat question in system prompt: 'Question: {question}'",
                "Add direct-answer instruction: 'Start your answer with the answer itself'",
            ]
        },
        'P4':  {
            'name':  'Retrieval Failure',
            'desc':  'Wrong documents retrieved — context is not relevant to the question',
            'mitigations': [
                "CRITICAL: Fix retrieval — improve embedding model",
                "Increase top-K to retrieve more candidate documents",
                "Add query expansion or query rewriting step",
                "Evaluate chunking strategy (smaller chunks = more targeted retrieval)",
                "Tune similarity threshold",
            ]
        },
        'P5':  {
            'name':  'Generation Failure',
            'desc':  'Retrieval worked (relevant context found) but model failed to extract correct answer',
            'mitigations': [
                "Improve extraction prompt: 'Extract the specific answer to the question'",
                "Add question-focusing instruction at top of prompt",
                "Provide few-shot examples of correct extraction",
                "Try more capable generation model",
            ]
        },
        'P6':  {
            'name':  'Unnecessary Verbosity',
            'desc':  'Answer is faithful and grounded but far longer than needed',
            'mitigations': [
                "Add hard length constraint: 'Answer in maximum 2 sentences'",
                "Use Concise or Baseline prompt variant",
                "Add few-shot examples with concise correct answers",
                "Lower temperature for more focused output",
            ]
        },
        'P7':  {
            'name':  'Complete Failure / Hallucination',
            'desc':  'All metrics critically low — model hallucinated a wrong, ungrounded answer',
            'mitigations': [
                "CRITICAL: Increase refusal threshold — model should say 'not available'",
                "Add hard grounding instruction: 'If not in documents, say Information not available'",
                "Require citations (forces model to ground claims)",
                "Fix retrieval if context relevancy also low",
                "Lower temperature to reduce creative hallucination",
            ]
        },
        'P8':  {
            'name':  'Presentation / Filler Issue',
            'desc':  'Correct content but poor presentation — filler phrases, qualifiers, noise',
            'mitigations': [
                "Prompt: 'State the answer directly without qualifiers'",
                "Remove meta-commentary: ban phrases like 'I think', 'it seems', 'Based on'",
                "Instruction: 'Answer as if writing a reference document'",
                "Post-process: strip known filler patterns",
            ]
        },
        'P9':  {
            'name':  'Retrieval-Driven Refusal',
            'desc':  'Model correctly refusing because retrieval returned insufficient '
                     'or irrelevant context (context relevancy < 0.65)',
            'mitigations': [
                "Fix retrieval (primary issue — not a generation problem)",
                "Increase top-K to retrieve more candidate documents",
                "Improve embedding model or chunking strategy",
                "Check if document corpus covers the question domain",
                "Add query expansion or rewriting step",
            ]
        },
        'P10': {
            'name':  'Over-Conservative Refusal',
            'desc':  'Model refusing even when clearly relevant context is available '
                     '(context relevancy ≥ 0.65)',
            'mitigations': [
                "Reduce refusal prompting: remove 'be cautious' instructions",
                "Add encouragement: 'Extract the answer from the documents provided'",
                "Lower refusal threshold in prompt",
                "Provide examples of valid extractions from context",
            ]
        },
        'P11': {
            'name':  'Citation Format Failure',
            'desc':  'Citation variant not producing required source attribution',
            'mitigations': [
                "Strengthen format instruction: 'ALWAYS cite in format: Answer: X Sources: Doc Y'",
                "Move citation instruction to start of prompt",
                "Add few-shot examples with perfect citation format",
                "Add post-processing to extract implicit citations from answer text",
            ]
        },
        'P12': {
            'name':  'Meta-Commentary Noise',
            'desc':  'Answer grounded but contains excessive filler/meta text reducing SNR',
            'mitigations': [
                "Prompt: 'State facts directly without preamble'",
                "Ban meta phrases: 'According to', 'The document says', 'I can see that'",
                "Instruction: 'Do not reference the documents by name in your answer'",
                "Post-process: strip common filler patterns",
            ]
        },
        'P13': {
            'name':  'Cascade Escalation Unresolved',
            'desc':  'Completeness cascade Tier 2 (LLM judge) triggered but answer still low quality',
            'mitigations': [
                "Review cascade trigger criteria — may be firing on correct low-completeness answers",
                "Check if question requires multi-hop reasoning beyond single-answer capability",
                "Consider whether gold answer captures full expected completeness",
                "Flag for human review — cascade cannot rescue fundamentally incomplete answers",
            ]
        },
        'P14': {
            'name':  'Cascade T1/T2 Conflict',
            'desc':  'Embedding (Tier 1) and LLM judge (Tier 2) completeness scores '
                     'genuinely conflict — both indicate poor quality',
            'mitigations': [
                "Review Tier 2 LLM judge prompt — may be too lenient or strict",
                "Check Tier 1 relevance_threshold — may be filtering too aggressively",
                "Flag for human review to determine ground truth",
                "Adjust cross_metric_delta trigger threshold if systematic",
            ]
        },
        'P15': {
            'name':  'Embedding Metric Artifact',
            'desc':  'Answer is factually correct (good F1 + faithfulness) but quality '
                     'score is pulled down by embedding metrics that underestimate '
                     'short answers — not a genuine quality failure',
            'mitigations': [
                "No prompt change needed — this is a metric limitation, not a model failure",
                "Document in evaluation report: quality score unreliable for ≤4 word answers",
                "Use F1 + Faithfulness as primary quality signal for short-answer QA",
                "Consider excluding from quality score calculation for factoid QA use cases",
            ]
        },
        'P16': {
            'name':  'Correct but Buried Answer',
            'desc':  'Answer contains the correct fact but buries it in a verbose paragraph — '
                     'common in Detailed variant. F1 near zero because key entity is diluted '
                     'by surrounding context text.',
            'mitigations': [
                "Add to Detailed prompt: 'State the answer in the first sentence, then elaborate'",
                "Post-process: extract first named entity from Detailed responses as clean answer",
                "Add few-shot examples showing 'Answer first, context second' format",
                "Consider switching to Citation or Baseline for factoid QA use cases",
            ]
        },
        'UNK': {
            'name':  'Unclassified',
            'desc':  'Failed quality thresholds but does not match any known rule-based pattern. '
                     'Passed to Layer 2 (cluster analysis) for grouping.',
            'mitigations': [
                "Review cluster assignment from Layer 2 analysis",
                "Check raw_answer for unexpected model behavior",
                "If cluster is coherent, consider promoting to a new named pattern",
            ]
        }
    }

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self, results_file: str = None):
        """
        Load evaluation results. If results_file is None, loads the most
        recent file from the results/ directory automatically.

        Args:
            results_file: Path to JSON results file, or None for auto-detect
        """
        if results_file is None:
            inv_files = sorted(
                glob.glob('results/multi_prompt_eval_results_*.json'),
                key=os.path.getmtime
            )
            if not inv_files:
                raise FileNotFoundError("No results files found in 'results/'")
            results_file = inv_files[-1]
            print(f"📂 Auto-loaded: {results_file}")
        else:
            print(f"📂 Loading: {results_file}")

        with open(results_file, 'r') as f:
            inv_data = json.load(f)

        self.results_file     = results_file
        self.detailed_results = inv_data['detailed_results']       # ← Fixed key
        self.aggregated       = inv_data['aggregated_results']     # ← Fixed key
        self.metadata         = inv_data['metadata']
        self.variants         = list(self.aggregated.keys())
        self.n                = self.metadata['num_questions']

        print(f"✅ Loaded {self.n} questions | {len(self.variants)} variants: "
              f"{', '.join(self.variants)}")

    # =========================================================================
    # PATTERN MATCHING
    # =========================================================================

    @staticmethod
    def _diagnose_pattern(vr: Dict,
                          variant_name: str,
                          hit_rate: float) -> Tuple[str, str]:
        """
        Rule-based diagnostic pattern matching.

        Maps metric combinations to one of 14 diagnostic patterns.
        Follows the decision logic from the diagnostic patterns guide.

        Args:
            vr:           Variant result dict (per-question, per-variant)
            variant_name: Prompt variant name (e.g. 'citation')
            hit_rate:     Retrieval hit rate for this question

        Returns:
            Tuple of (pattern_code, one_sentence_rationale)
        """
        # Extract metrics with safe defaults
        f1       = vr.get('f1_score', 0) or 0
        gnd      = vr.get('groundedness', 0) or 0
        faith    = vr.get('faithfulness', 0) or 0
        ans_rel  = vr.get('answer_relevancy', 0) or 0
        ctx_rel  = vr.get('context_relevancy', 0) or 0
        concise  = vr.get('conciseness', 0) or 0
        snr      = vr.get('relevance_snr', 0) or 0
        quality  = vr.get('quality_score', 0) or 0
        is_refusal   = vr.get('is_refusal', False)
        has_citation = vr.get('has_citation', False)
        t2_triggered = vr.get('tier2_triggered', False)
        agreement    = vr.get('agreement')          # bool or None
        ctx_cov  = vr.get('context_coverage', 0) or 0
        completeness = vr.get('completeness', 0) or 0

        # ---- P9: Retrieval-driven refusal (MUST come before P7) ----
        # Refusals always have low F1/groundedness/faithfulness — if checked
        # after P7, they get misclassified as hallucinations. Check first.
        if is_refusal and ctx_rel < 0.50:
            return ('P9',
                    f"Model refusing — likely because retrieval returned insufficient "
                    f"context (context relevancy={ctx_rel:.2f}, hit rate={hit_rate:.0%})")

        # ---- P10: Over-conservative refusal (MUST come before P7) ----
        if is_refusal and ctx_rel >= 0.50:
            return ('P10',
                    f"Model refusing despite relevant context available "
                    f"(context relevancy={ctx_rel:.2f}) — model too conservative")

        # ---- P13: Cascade escalated but completeness still low ----
        if t2_triggered and (completeness or 0) < 0.4:
            return ('P13',
                    f"Tier 2 LLM judge triggered (cascade) but final completeness "
                    f"still low ({completeness:.2f}) — cascade unable to rescue answer")

        # ---- P14: T1/T2 genuine conflict (tightened condition) ----
        # Only flag when BOTH T1 and T2 disagree AND T2 also scores answer poorly.
        # If T2 scores high but T1 is low, that is the EXPECTED embedding limitation
        # for short answers — not a genuine conflict worth human review.
        if t2_triggered and agreement is False and ctx_cov is not None:
            delta = abs(ctx_cov - (completeness or 0))
            if delta > 0.4 and (completeness or 0) < 0.5:
                return ('P14',
                        f"Genuine T1/T2 conflict: Tier 1 ({ctx_cov:.2f}) and "
                        f"Tier 2 ({completeness:.2f}) both indicate issues "
                        f"(Δ={delta:.2f}) — manual review needed")

        # ---- P7: Complete failure / hallucination (non-refusal only) ----
        # Guard: is_refusal checked above. P7 only fires for actual wrong answers,
        # not model refusals which look identical on low-metric surface.
        if not is_refusal and f1 < 0.3 and gnd < 0.4 and faith < 0.4:
            return ('P7',
                    f"All metrics critically low — likely hallucinated wrong answer "
                    f"(F1={f1:.2f}, Ground={gnd:.2f}, Faith={faith:.2f})")

        # ---- P4: Retrieval failure (non-refusal) ----
        if not is_refusal and ctx_rel < 0.35 and gnd < 0.45:
            return ('P4',
                    f"Low context relevancy ({ctx_rel:.2f}) and low groundedness "
                    f"({gnd:.2f}) indicate retrieval failure — wrong docs retrieved")

        # ---- P1: High F1 but hallucinated details ----
        if f1 >= 0.5 and gnd < 0.45 and faith < 0.55:
            return ('P1',
                    f"Correct answer (F1={f1:.2f}) but ungrounded supporting details "
                    f"(Ground={gnd:.2f}, Faith={faith:.2f})")

        # ---- P2: Low F1 but grounded/faithful → verbose or off-target ----
        if f1 < 0.4 and gnd >= 0.55 and faith >= 0.55:
            return ('P2',
                    f"Grounded ({gnd:.2f}) and faithful ({faith:.2f}) but wrong format "
                    f"— likely too verbose or off-target (F1={f1:.2f})")

        # ---- P5: Good retrieval but generation failed ----
        if ctx_rel >= 0.55 and ans_rel < 0.35 and not is_refusal:
            return ('P5',
                    f"Relevant context retrieved ({ctx_rel:.2f}) but answer doesn't "
                    f"address question (answer relevancy={ans_rel:.2f})")

        # ---- P3: Tangential answer (long answers only) ----
        # Guard: skip short answers (≤4 words) — ans_rel is structurally low
        # for 2-word factoid answers vs long questions due to embedding scaling.
        # Only flag as tangential when the answer is long enough for ans_rel
        # to be a meaningful signal.
        inv_answer_words = len((vr.get('clean_answer') or '').split())
        if f1 >= 0.5 and ans_rel < 0.35 and inv_answer_words >= 5:
            return ('P3',
                    f"F1 acceptable ({f1:.2f}) but answer tangential to question "
                    f"(answer relevancy={ans_rel:.2f}, answer length={inv_answer_words}w)")

        # ---- P11: Citation variant not citing ----
        if variant_name == 'citation' and not has_citation and not is_refusal:
            return ('P11',
                    f"Citation prompt variant failed to produce source attribution")

        # ---- P12: High SNR but meta-commentary noise ----
        if gnd >= 0.60 and snr < 0.40:
            return ('P12',
                    f"Grounded ({gnd:.2f}) but low signal-to-noise ratio ({snr:.2f}) "
                    f"— likely meta-commentary or filler phrases")

        # ---- P6: Verbose but correct ----
        if faith >= 0.65 and concise < 0.35:
            return ('P6',
                    f"Faithful ({faith:.2f}) but unnecessarily verbose "
                    f"(conciseness={concise:.2f})")

        # ---- P8: Poor presentation despite good components ----
        if quality < 0.55 and f1 >= 0.5 and faith >= 0.60:
            return ('P8',
                    f"Good F1 ({f1:.2f}) and faithfulness ({faith:.2f}) but low "
                    f"quality score ({quality:.2f}) — likely filler/presentation issue")

        # ---- P16: Correct but buried answer (Cluster 0 promoted pattern) ----
        # Verbose answers (≥30 words) where F1 is near zero despite high
        # groundedness and answer relevancy — the correct fact exists but is
        # diluted by surrounding context text. Concentrated in Detailed variant.
        if (not is_refusal and f1 < 0.2 and gnd >= 0.55 and
                ans_rel >= 0.55 and inv_answer_words >= 30):
            return ('P16',
                    f"Correct fact buried in verbose paragraph — F1={f1:.2f} despite "
                    f"Ground={gnd:.2f}, AnsRel={ans_rel:.2f}, length={inv_answer_words}w")

        # ---- P15: Embedding metric artifact (NEW) ----
        # Answer is factually correct (good F1 + faithfulness) but quality score
        # is pulled down by embedding metrics that systematically underestimate
        # short answers (ans_rel, conciseness, SNR all scale poorly for ≤4 words).
        # This is a metric limitation, not a genuine model quality failure.
        if f1 >= 0.5 and faith >= 0.65 and quality < 0.55:
            return ('P15',
                    f"Correct answer (F1={f1:.2f}) and faithful ({faith:.2f}) "
                    f"but quality score ({quality:.2f}) pulled down by embedding "
                    f"metric artifacts — likely short-answer scaling issue")

        return ('UNK', f"Failed thresholds but no matching pattern "
                       f"(F1={f1:.2f}, Ground={gnd:.2f}, Faith={faith:.2f}, "
                       f"Quality={quality:.2f}) — passed to Layer 2 cluster analysis")

    # =========================================================================
    # FIND LOW QUALITY ANSWERS
    # =========================================================================

    def find_low_quality_answers(self,
                                  f1_threshold: float = 0.3,
                                  groundedness_threshold: float = 0.5,
                                  faithfulness_threshold: float = 0.5,
                                  completeness_threshold: float = 0.4,
                                  quality_score_threshold: float = 0.55,
                                  variants: List[str] = None,
                                  include_patterns: List[str] = None
                                  ) -> List[Dict]:
        """
        Identify answers that fail any quality threshold and classify them
        into diagnostic patterns.

        Whitelist guard (applied first):
          Answers with F1 ≥ 0.5 AND quality_score ≥ 0.70 are skipped
          regardless of individual metric values. This prevents borderline
          groundedness on correct short answers (an embedding artifact) from
          generating false positives. Cluster analysis confirmed these cases
          are not genuine failures.

        An answer is flagged if it passes the whitelist AND any of:
          - F1 < f1_threshold
          - groundedness < groundedness_threshold
          - faithfulness < faithfulness_threshold
          - completeness < completeness_threshold (cascade final)
          - quality_score < quality_score_threshold
          - Citation variant missing citation (not a refusal)
          - Cascade escalated but completeness still low
          - T1/T2 conflict with large disagreement

        Results are sorted by severity (worst first).

        Args:
            f1_threshold:            F1 below this = quality issue
            groundedness_threshold:  Groundedness below this = grounding issue
            faithfulness_threshold:  Faithfulness below this = faithfulness issue
            completeness_threshold:  Completeness (cascade) below this = issue
            quality_score_threshold: Quality score below this = overall issue
            variants:                Filter to specific variants (None = all)
            include_patterns:        Filter to specific pattern codes (None = all)

        Returns:
            List of flagged result dicts, sorted by severity descending
        """
        inv_target_variants = variants or self.variants
        inv_flagged = []

        for inv_i, inv_result in enumerate(self.detailed_results):
            inv_question  = inv_result['question']
            inv_gold      = inv_result['gold_answer']
            inv_hit_rate  = inv_result.get('hit_rate', 0)  # ← Fixed key

            for inv_variant in inv_target_variants:
                inv_vr = inv_result['variant_results'].get(inv_variant)  # ← Fixed key
                if inv_vr is None or inv_vr.get('is_error', False):
                    continue

                # ── WHITELIST GUARD ────────────────────────────────────────
                # Skip answers that are clearly correct and high overall quality.
                # Without this, borderline groundedness (e.g. 0.48) on a correct
                # short answer (F1=1.0, quality=0.80) generates false positives —
                # the answer is fine; the threshold is catching embedding noise.
                # Condition: correct answer (F1 ≥ 0.5) + good quality (≥ 0.70)
                # → not a genuine quality failure regardless of individual metric
                inv_f1_pre    = inv_vr.get('f1_score', 0) or 0
                inv_qual_pre  = inv_vr.get('quality_score') or 0
                if inv_f1_pre >= 0.5 and inv_qual_pre >= 0.70:
                    continue  # Pass — correct answer with good composite quality

                # Collect all quality issues for this answer
                inv_issues = []
                inv_severity = 0.0

                # --- Correctness ---
                inv_f1 = inv_vr.get('f1_score', 0) or 0
                if inv_f1 < f1_threshold:
                    inv_issues.append(f"Low F1: {inv_f1:.3f} (threshold {f1_threshold})")
                    inv_severity += (f1_threshold - inv_f1) * 2.0  # Weight higher

                # --- Groundedness ---
                inv_gnd = inv_vr.get('groundedness')
                if inv_gnd is not None and inv_gnd < groundedness_threshold:
                    inv_issues.append(f"Low Groundedness: {inv_gnd:.3f} "
                                      f"(threshold {groundedness_threshold})")
                    inv_severity += (groundedness_threshold - inv_gnd) * 1.5

                # --- Faithfulness ---
                inv_faith = inv_vr.get('faithfulness')
                if inv_faith is not None and inv_faith < faithfulness_threshold:
                    inv_issues.append(f"Low Faithfulness: {inv_faith:.3f} "
                                      f"(threshold {faithfulness_threshold})")
                    inv_severity += (faithfulness_threshold - inv_faith) * 1.5

                # --- Completeness (cascade final) ---
                inv_comp = inv_vr.get('completeness')
                if inv_comp is not None and inv_comp < completeness_threshold:
                    inv_issues.append(f"Low Completeness: {inv_comp:.3f} "
                                      f"(threshold {completeness_threshold})")
                    inv_severity += (completeness_threshold - inv_comp)

                # --- Quality score ---
                inv_qual = inv_vr.get('quality_score')
                if inv_qual is not None and inv_qual < quality_score_threshold:
                    inv_issues.append(f"Low Quality Score: {inv_qual:.3f} "
                                      f"(threshold {quality_score_threshold})")
                    inv_severity += (quality_score_threshold - inv_qual)

                # --- Citation compliance ---
                if (inv_variant == 'citation' and
                        not inv_vr.get('has_citation', False) and   # ← Fixed key
                        not inv_vr.get('is_refusal', False)):
                    inv_issues.append("Citation variant: missing source attribution")
                    inv_severity += 0.4

                # --- Cascade-specific issues ---
                if inv_vr.get('tier2_triggered') and (inv_comp or 0) < completeness_threshold:
                    inv_issues.append(f"Cascade escalated but completeness still low "
                                      f"({inv_comp:.3f})")
                    inv_severity += 0.3

                if (inv_vr.get('tier2_triggered') and
                        inv_vr.get('agreement') is False):
                    inv_ctx_cov = inv_vr.get('context_coverage') or 0
                    inv_delta   = abs(inv_ctx_cov - (inv_comp or 0))
                    if inv_delta > 0.4:
                        inv_issues.append(f"Cascade T1/T2 conflict: "
                                          f"T1={inv_ctx_cov:.2f} vs T2={inv_comp:.2f} "
                                          f"(Δ={inv_delta:.2f})")
                        inv_severity += 0.3

                if not inv_issues:
                    continue  # Answer passes all thresholds — skip

                # --- Pattern classification ---
                inv_pattern_code, inv_rationale = LowQualityInvestigator._diagnose_pattern(
                    vr=inv_vr,
                    variant_name=inv_variant,
                    hit_rate=inv_hit_rate
                )

                # --- Filter by pattern if requested ---
                if include_patterns and inv_pattern_code not in include_patterns:
                    continue

                inv_flagged.append({
                    'question_id':       inv_i + 1,
                    'variant':           inv_variant,
                    'question':          inv_question,
                    'gold_answer':       inv_gold,
                    'generated_answer':  inv_vr.get('clean_answer', ''),
                    'hit_rate':          inv_hit_rate,
                    # Correctness
                    'f1_score':          inv_f1,
                    'correct_f1':        inv_vr.get('correct_f1', False),
                    'contains':          inv_vr.get('correct_contains', 0),
                    'is_refusal':        inv_vr.get('is_refusal', False),
                    'has_citation':      inv_vr.get('has_citation', False),  # ← Fixed key
                    'answer_length':     inv_vr.get('answer_length', 0),
                    # Quality metrics
                    'groundedness':      inv_vr.get('groundedness'),
                    'context_coverage':  inv_vr.get('context_coverage'),   # Tier 1
                    'completeness':      inv_vr.get('completeness'),        # Cascade final
                    'faithfulness':      inv_vr.get('faithfulness'),
                    'answer_relevancy':  inv_vr.get('answer_relevancy'),
                    'context_relevancy': inv_vr.get('context_relevancy'),
                    'conciseness':       inv_vr.get('conciseness'),
                    'relevance_snr':     inv_vr.get('relevance_snr'),
                    'quality_score':     inv_vr.get('quality_score'),
                    # Cascade
                    'tier2_triggered':        inv_vr.get('tier2_triggered', False),
                    'tier2_trigger_reasons':  inv_vr.get('tier2_trigger_reasons', []),
                    'agreement':              inv_vr.get('agreement'),
                    # Diagnosis
                    'pattern_code':      inv_pattern_code,
                    'pattern_name':      LowQualityInvestigator.PATTERN_DEFINITIONS[
                                             inv_pattern_code]['name'],
                    'pattern_rationale': inv_rationale,
                    'issues':            inv_issues,
                    'severity_score':    round(inv_severity, 4),
                    # For audit trail
                    'quality_issues':    inv_vr.get('quality_issues', []),
                })

        # Sort by severity descending
        inv_flagged.sort(key=lambda x: x['severity_score'], reverse=True)
        return inv_flagged

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_low_quality_report(self,
                                   inv_flagged: List[Dict],
                                   output_dir: str = 'results/low_quality_analysis',
                                   top_worst_cases: int = 20) -> Dict:
        """
        Export low-quality findings for human review.

        Creates:
          1. low_quality_summary.csv        — all flagged answers (spreadsheet)
          2. low_quality_detailed.json      — full data with all metrics
          3. pattern_summary.json           — pattern frequency + mitigations
          4. worst_cases/case_NN_*.md       — individual markdown per worst case

        Args:
            inv_flagged:      Output from find_low_quality_answers()
            output_dir:       Directory to write outputs
            top_worst_cases:  Number of individual markdown files to generate

        Returns:
            Dict with output file paths and summary statistics
        """
        inv_out = Path(output_dir)
        inv_out.mkdir(parents=True, exist_ok=True)

        if not inv_flagged:
            print("✅ No low-quality answers found at current thresholds.")
            return {'total_flagged': 0}

        # ------------------------------------------------------------------
        # 1. Summary CSV
        # ------------------------------------------------------------------
        inv_csv_file = inv_out / 'low_quality_summary.csv'
        inv_csv_rows = []

        for inv_item in inv_flagged:
            def inv_fmt(val):
                return f"{val:.3f}" if isinstance(val, float) else (str(val) if val is not None else 'N/A')

            inv_csv_rows.append({
                'Question ID':        inv_item['question_id'],
                'Variant':            inv_item['variant'],
                'Pattern Code':       inv_item['pattern_code'],
                'Pattern Name':       inv_item['pattern_name'],
                'Severity':           f"{inv_item['severity_score']:.3f}",
                'Question':           inv_item['question'][:120],
                'Gold Answer':        inv_item['gold_answer'],
                'Generated Answer':   inv_item['generated_answer'][:120],
                'F1':                 inv_fmt(inv_item['f1_score']),
                'Groundedness':       inv_fmt(inv_item['groundedness']),
                'Faithfulness':       inv_fmt(inv_item['faithfulness']),
                'Context Coverage':   inv_fmt(inv_item['context_coverage']),
                'Completeness':       inv_fmt(inv_item['completeness']),
                'Answer Relevancy':   inv_fmt(inv_item['answer_relevancy']),
                'Context Relevancy':  inv_fmt(inv_item['context_relevancy']),
                'Conciseness':        inv_fmt(inv_item['conciseness']),
                'SNR':                inv_fmt(inv_item['relevance_snr']),
                'Quality Score':      inv_fmt(inv_item['quality_score']),
                'Hit Rate':           f"{inv_item['hit_rate']:.1%}",
                'Is Refusal':         inv_item['is_refusal'],
                'Has Citation':       inv_item['has_citation'],
                'T2 Triggered':       inv_item['tier2_triggered'],
                'T1/T2 Agreement':    inv_item['agreement'],
                'Issues':             ' | '.join(inv_item['issues']),
                'Pattern Rationale':  inv_item['pattern_rationale'],
            })

        if inv_csv_rows:
            inv_fieldnames = list(inv_csv_rows[0].keys())
            with open(inv_csv_file, 'w', newline='', encoding='utf-8') as f:
                inv_writer = csv.DictWriter(f, fieldnames=inv_fieldnames)
                inv_writer.writeheader()
                inv_writer.writerows(inv_csv_rows)
            print(f"✓ Summary CSV: {inv_csv_file}  ({len(inv_flagged)} rows)")

        # ------------------------------------------------------------------
        # 2. Detailed JSON
        # ------------------------------------------------------------------
        inv_json_file = inv_out / 'low_quality_detailed.json'
        inv_export_items = [
            {k: v for k, v in item.items()} for item in inv_flagged
        ]
        with open(inv_json_file, 'w') as f:
            json.dump(inv_export_items, f, indent=2, default=str)
        print(f"✓ Detailed JSON: {inv_json_file}")

        # ------------------------------------------------------------------
        # 3. Pattern summary JSON
        # ------------------------------------------------------------------
        inv_pattern_counts = defaultdict(int)
        inv_pattern_severity = defaultdict(list)

        for item in inv_flagged:
            inv_pattern_counts[item['pattern_code']] += 1
            inv_pattern_severity[item['pattern_code']].append(item['severity_score'])

        inv_pattern_summary = {}
        for code, count in sorted(inv_pattern_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            defn = LowQualityInvestigator.PATTERN_DEFINITIONS.get(code, {})
            inv_pattern_summary[code] = {
                'pattern_name':    defn.get('name', code),
                'description':     defn.get('desc', ''),
                'count':           count,
                'pct_of_flagged':  count / len(inv_flagged),
                'avg_severity':    float(np.mean(inv_pattern_severity[code])),
                'max_severity':    float(np.max(inv_pattern_severity[code])),
                'mitigations':     defn.get('mitigations', []),
            }

        inv_psummary_file = inv_out / 'pattern_summary.json'
        with open(inv_psummary_file, 'w') as f:
            json.dump(inv_pattern_summary, f, indent=2)
        print(f"✓ Pattern summary: {inv_psummary_file}")

        # ------------------------------------------------------------------
        # 4. Individual worst-case markdown files
        # ------------------------------------------------------------------
        inv_worst_dir = inv_out / 'worst_cases'
        inv_worst_dir.mkdir(exist_ok=True)

        for inv_rank, inv_item in enumerate(inv_flagged[:top_worst_cases], 1):
            inv_md = self._generate_markdown_report(inv_item, inv_rank)
            inv_md_name = (f"case_{inv_rank:02d}_q{inv_item['question_id']}_"
                           f"{inv_item['variant']}_{inv_item['pattern_code']}.md")
            inv_md_file = inv_worst_dir / inv_md_name
            with open(inv_md_file, 'w', encoding='utf-8') as f:
                f.write(inv_md)

        print(f"✓ Top {min(top_worst_cases, len(inv_flagged))} worst cases: {inv_worst_dir}/")

        return {
            'csv_file':          str(inv_csv_file),
            'json_file':         str(inv_json_file),
            'pattern_summary':   str(inv_psummary_file),
            'worst_cases_dir':   str(inv_worst_dir),
            'total_flagged':     len(inv_flagged),
            'pattern_breakdown': {k: v['count'] for k, v in inv_pattern_summary.items()},
        }

    # =========================================================================
    # MARKDOWN REPORT (PER CASE)
    # =========================================================================

    def _generate_markdown_report(self, inv_item: Dict, rank: int) -> str:
        """Generate a detailed markdown report for a single flagged case."""

        def mfmt(val, pct=False):
            if val is None:
                return 'N/A'
            if pct:
                return f'{val:.1%}'
            return f'{val:.3f}'

        defn = LowQualityInvestigator.PATTERN_DEFINITIONS.get(
            inv_item['pattern_code'], {}
        )

        inv_md = f"""# Low Quality Case #{rank} — {inv_item['pattern_name']}

## Case Overview

| Field | Value |
|-------|-------|
| **Question ID** | {inv_item['question_id']} |
| **Variant** | {inv_item['variant'].capitalize()} |
| **Pattern** | `{inv_item['pattern_code']}` — {inv_item['pattern_name']} |
| **Severity Score** | {inv_item['severity_score']:.3f} |
| **Retrieval Hit Rate** | {mfmt(inv_item['hit_rate'], pct=True)} |

---

## Content

**Question:**
> {inv_item['question']}

**Gold Answer:** `{inv_item['gold_answer']}`

**Generated Answer:** `{inv_item['generated_answer']}`

---

## Diagnostic

### Pattern Identified: `{inv_item['pattern_code']}` — {inv_item['pattern_name']}

**Description:** {defn.get('desc', 'N/A')}

**Rationale for this case:** {inv_item['pattern_rationale']}

### Issues Detected

"""
        for issue in inv_item['issues']:
            inv_md += f"- ❌ {issue}\n"

        inv_md += f"""
---

## Metrics

### Correctness

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| F1 Score | {mfmt(inv_item['f1_score'])} | ≥0.50 | {'✅' if (inv_item['f1_score'] or 0) >= 0.5 else '⚠️'} |
| Contains | {mfmt(inv_item['contains'])} | — | — |
| Is Refusal | {inv_item['is_refusal']} | — | — |
| Has Citation | {inv_item['has_citation']} | — | — |
| Answer Length | {inv_item['answer_length']} words | — | — |

### Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Groundedness | {mfmt(inv_item['groundedness'])} | ≥0.50 | {'✅' if (inv_item['groundedness'] or 0) >= 0.5 else '⚠️'} |
| Context Coverage (T1) | {mfmt(inv_item['context_coverage'])} | ≥0.40 | {'✅' if (inv_item['context_coverage'] or 0) >= 0.4 else '⚠️'} |
| Completeness (Cascade) | {mfmt(inv_item['completeness'])} | ≥0.40 | {'✅' if (inv_item['completeness'] or 0) >= 0.4 else '⚠️'} |
| Faithfulness | {mfmt(inv_item['faithfulness'])} | ≥0.70 | {'✅' if (inv_item['faithfulness'] or 0) >= 0.7 else '⚠️'} |
| Answer Relevancy | {mfmt(inv_item['answer_relevancy'])} | ≥0.50 | {'✅' if (inv_item['answer_relevancy'] or 0) >= 0.5 else '⚠️'} |
| Context Relevancy | {mfmt(inv_item['context_relevancy'])} | ≥0.40 | {'✅' if (inv_item['context_relevancy'] or 0) >= 0.4 else '⚠️'} |
| Conciseness | {mfmt(inv_item['conciseness'])} | ≥0.50 | {'✅' if (inv_item['conciseness'] or 0) >= 0.5 else '⚠️'} |
| Relevance SNR | {mfmt(inv_item['relevance_snr'])} | ≥0.70 | {'✅' if (inv_item['relevance_snr'] or 0) >= 0.7 else '⚠️'} |
| Quality Score | {mfmt(inv_item['quality_score'])} | ≥0.70 | {'✅' if (inv_item['quality_score'] or 0) >= 0.7 else '⚠️'} |

"""

        # Cascade section (only if triggered)
        if inv_item['tier2_triggered']:
            inv_md += f"""### Completeness Cascade (Tier 2 Triggered)

| Field | Value |
|-------|-------|
| Context Coverage (T1 embedding) | {mfmt(inv_item['context_coverage'])} |
| Completeness (T2 LLM judge) | {mfmt(inv_item['completeness'])} |
| T1/T2 Agreement | {inv_item['agreement']} |

**Trigger reasons:**
"""
            for reason in (inv_item['tier2_trigger_reasons'] or []):
                inv_md += f"- {reason}\n"
            inv_md += "\n"

        inv_md += f"""---

## Recommended Actions

"""
        for mitigation in defn.get('mitigations', ['Manual review recommended']):
            inv_md += f"- ✅ {mitigation}\n"

        # Add quality guard issues if present
        if inv_item.get('quality_issues'):
            inv_md += f"""
### Quality Guard Flags

"""
            for flag in inv_item['quality_issues']:
                inv_md += f"- {flag}\n"

        inv_md += f"""
---

*Generated by LLM Evaluation Framework — Low Quality Investigator*
*Pattern: `{inv_item['pattern_code']}` | Severity: {inv_item['severity_score']:.3f}*
"""
        return inv_md

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    def print_pattern_summary(self, inv_flagged: List[Dict]) -> None:
        """Print pattern frequency breakdown to console."""
        if not inv_flagged:
            print("✅ No low-quality answers found.")
            return

        inv_pattern_counts = defaultdict(int)
        inv_pattern_severity = defaultdict(list)
        inv_variant_counts = defaultdict(lambda: defaultdict(int))

        for item in inv_flagged:
            inv_pattern_counts[item['pattern_code']] += 1
            inv_pattern_severity[item['pattern_code']].append(item['severity_score'])
            inv_variant_counts[item['variant']][item['pattern_code']] += 1

        print("\n" + "="*80)
        print("📊 LOW QUALITY ANSWER PATTERN SUMMARY")
        print("="*80)
        print(f"\nTotal flagged: {len(inv_flagged)} / "
              f"{self.n * len(self.variants)} "
              f"({len(inv_flagged) / (self.n * len(self.variants)):.1%})\n")

        print(f"{'Code':<6} {'Pattern Name':<35} {'Count':>6} {'%':>6} "
              f"{'AvgSev':>8} {'MaxSev':>8}")
        print("-" * 75)

        for code, count in sorted(inv_pattern_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            defn = LowQualityInvestigator.PATTERN_DEFINITIONS.get(code, {})
            name = defn.get('name', code)[:33]
            pct  = count / len(inv_flagged) * 100
            avg_sev = np.mean(inv_pattern_severity[code])
            max_sev = np.max(inv_pattern_severity[code])
            print(f"{code:<6} {name:<35} {count:>6} {pct:>5.1f}% "
                  f"{avg_sev:>8.3f} {max_sev:>8.3f}")

        print("\n" + "─"*60)
        print("Pattern breakdown by variant:")
        print(f"\n{'Variant':<12}", end='')
        all_codes = sorted(inv_pattern_counts.keys())
        for code in all_codes:
            print(f"  {code:<6}", end='')
        print()
        print("-" * (12 + len(all_codes) * 8))
        for variant in self.variants:
            print(f"{variant.capitalize():<12}", end='')
            for code in all_codes:
                cnt = inv_variant_counts[variant][code]
                print(f"  {cnt:<6}", end='')
            print()

        print("\n" + "─"*60)
        print("Top 3 patterns by frequency — recommended actions:\n")
        for code, count in sorted(inv_pattern_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:3]:
            defn = LowQualityInvestigator.PATTERN_DEFINITIONS.get(code, {})
            print(f"  [{code}] {defn.get('name', code)} ({count} cases):")
            for m in defn.get('mitigations', [])[:2]:
                print(f"    → {m}")
            print()

        print("="*80)

    def print_worst_cases(self, inv_flagged: List[Dict], num_cases: int = 10) -> None:
        """Print worst cases to console for quick review."""
        print("\n" + "="*80)
        print(f"🔍 TOP {num_cases} WORST QUALITY ANSWERS")
        print("="*80)

        for rank, item in enumerate(inv_flagged[:num_cases], 1):
            print(f"\n{'─'*70}")
            print(f"  #{rank} | Q{item['question_id']} | {item['variant'].capitalize()} | "
                  f"[{item['pattern_code']}] {item['pattern_name']} | "
                  f"Severity: {item['severity_score']:.3f}")
            print(f"{'─'*70}")
            print(f"  Q: {item['question'][:90]}...")
            print(f"  Gold:  '{item['gold_answer']}'")
            print(f"  Answer:'{item['generated_answer'][:80]}'")
            def _fmt2(val):
                return f"{val:.2f}" if val is not None else 'N/A'

            print(f"\n  Metrics: F1={item['f1_score']:.2f} | "
                  f"Ground={_fmt2(item['groundedness'])} | "
                  f"Faith={_fmt2(item['faithfulness'])} | "
                  f"Complete={_fmt2(item['completeness'])} | "
                  f"HitRate={item['hit_rate']:.0%}")
            if item['tier2_triggered']:
                print(f"  Cascade: T2 triggered | "
                      f"CtxCov={_fmt2(item['context_coverage'])} | "
                      f"Agreement={item['agreement']}")
            print(f"  Diagnosis: {item['pattern_rationale']}")
            print(f"  Issues: {' | '.join(item['issues'][:2])}")

        print("\n" + "="*80)


    # =========================================================================
    # LAYER 2: CLUSTER ANALYSIS (UNK CASES ONLY)
    # =========================================================================

    # Metric features used for clustering — all numeric, normalised to [0,1]
    _CLUSTER_FEATURES = [
        'f1_score', 'groundedness', 'faithfulness', 'completeness',
        'answer_relevancy', 'context_relevancy', 'conciseness',
        'relevance_snr', 'quality_score', 'hit_rate', 'answer_length',
    ]

    @staticmethod
    def _build_feature_matrix(inv_items: List[Dict]) -> np.ndarray:
        """
        Build a normalized numeric feature matrix for clustering.
        Missing values imputed with column median.
        answer_length is log-scaled to reduce outlier effect.
        """
        inv_rows = []
        for item in inv_items:
            inv_row = []
            for feat in LowQualityInvestigator._CLUSTER_FEATURES:
                val = item.get(feat) or 0.0
                if feat == 'answer_length':
                    val = np.log1p(val)   # log-scale length
                inv_row.append(float(val))
            inv_rows.append(inv_row)

        inv_X = np.array(inv_rows)

        # Impute column medians for any remaining zeros that might be missing
        for col in range(inv_X.shape[1]):
            col_median = np.median(inv_X[inv_X[:, col] != 0, col]) if np.any(inv_X[:, col] != 0) else 0
            inv_X[inv_X[:, col] == 0, col] = col_median

        inv_scaler = StandardScaler()
        return inv_scaler.fit_transform(inv_X)

    @staticmethod
    def _select_k(inv_X: np.ndarray,
                   k_min: int = 2,
                   k_max: int = 6) -> int:
        """
        Auto-select K for K-Means using silhouette score.
        Returns K with highest silhouette score in [k_min, k_max].
        Falls back to k_min if too few samples.
        """
        if len(inv_X) < k_min * 2:
            return k_min

        inv_best_k     = k_min
        inv_best_score = -1.0

        for k in range(k_min, min(k_max + 1, len(inv_X))):
            inv_km = KMeans(n_clusters=k, random_state=42, n_init=10)
            inv_labels = inv_km.fit_predict(inv_X)
            # Silhouette undefined for k=1 or all same label
            if len(set(inv_labels)) < 2:
                continue
            inv_score = silhouette_score(inv_X, inv_labels)
            if inv_score > inv_best_score:
                inv_best_score = inv_score
                inv_best_k     = k

        return inv_best_k

    def cluster_unk_answers(self,
                             inv_flagged: List[Dict],
                             k: int = None,
                             k_min: int = 2,
                             k_max: int = 6
                             ) -> Optional[Dict]:
        """
        Layer 2: Cluster UNK (unclassified) answers using K-Means on
        normalized metric feature vectors.

        Only runs on UNK cases — classified patterns are not mixed in.
        Auto-selects optimal K via silhouette score unless K is specified.

        Args:
            inv_flagged: Output from find_low_quality_answers()
            k:           Number of clusters (None = auto-select)
            k_min:       Minimum K for auto-selection (default 2)
            k_max:       Maximum K for auto-selection (default 6)

        Returns:
            Dict with cluster assignments and profiles, or None if
            fewer than 4 UNK cases (not worth clustering).
        """
        inv_unk = [item for item in inv_flagged if item['pattern_code'] == 'UNK']

        if len(inv_unk) < 4:
            print(f"⚠️  Only {len(inv_unk)} UNK cases — skipping cluster analysis "
                  f"(minimum 4 required)")
            return None

        print(f"\n🔬 Layer 2: Clustering {len(inv_unk)} UNK answers...")

        # Build feature matrix
        inv_X = LowQualityInvestigator._build_feature_matrix(inv_unk)

        # Select K
        if k is None:
            k = LowQualityInvestigator._select_k(inv_X, k_min, k_max)

        print(f"   K={k} clusters selected", end='')

        # Fit K-Means
        inv_km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        inv_labels = inv_km.fit_predict(inv_X)

        # Silhouette score
        if len(set(inv_labels)) >= 2:
            inv_sil = silhouette_score(inv_X, inv_labels)
            print(f" | Silhouette score: {inv_sil:.3f}")
        else:
            inv_sil = 0.0
            print()

        # Assign cluster labels back to items
        for item, label in zip(inv_unk, inv_labels):
            item['cluster_id'] = int(label)

        # Build cluster profiles
        inv_clusters = {}
        for cid in range(k):
            inv_cluster_items = [item for item in inv_unk
                                  if item.get('cluster_id') == cid]
            if not inv_cluster_items:
                continue

            # Metric averages
            inv_profile = {}
            for feat in LowQualityInvestigator._CLUSTER_FEATURES:
                vals = [item.get(feat) or 0.0 for item in inv_cluster_items]
                inv_profile[feat] = float(np.mean(vals))

            # Find centroid (item closest to cluster center in feature space)
            inv_cluster_X = inv_X[inv_labels == cid]
            inv_center    = inv_km.cluster_centers_[cid]
            inv_distances = np.linalg.norm(inv_cluster_X - inv_center, axis=1)
            inv_centroid_idx = int(np.argmin(inv_distances))
            inv_centroid_item = inv_cluster_items[inv_centroid_idx]

            # Variant distribution
            inv_variant_dist = defaultdict(int)
            for item in inv_cluster_items:
                inv_variant_dist[item['variant']] += 1

            # Auto-generated hypothesis based on dominant metric signals
            inv_hypothesis = LowQualityInvestigator._hypothesize_cluster(inv_profile)

            inv_clusters[cid] = {
                'cluster_id':      cid,
                'size':            len(inv_cluster_items),
                'pct_of_unk':      len(inv_cluster_items) / len(inv_unk),
                'metric_profile':  inv_profile,
                'variant_dist':    dict(inv_variant_dist),
                'centroid_item':   inv_centroid_item,
                'hypothesis':      inv_hypothesis,
                'items':           inv_cluster_items,
            }

        return {
            'k':              k,
            'n_unk':          len(inv_unk),
            'silhouette':     inv_sil,
            'clusters':       inv_clusters,
        }

    @staticmethod
    def _hypothesize_cluster(profile: Dict) -> str:
        """
        Generate a plain-English hypothesis about a cluster's failure mode
        based on its average metric profile.
        Used to help human reviewers quickly assess whether a cluster
        represents a genuine new pattern.
        """
        f1    = profile.get('f1_score', 0)
        gnd   = profile.get('groundedness', 0)
        faith = profile.get('faithfulness', 0)
        ans_r = profile.get('answer_relevancy', 0)
        ctx_r = profile.get('context_relevancy', 0)
        qual  = profile.get('quality_score', 0)
        length = min(profile.get('answer_length', 0), 9999)  # raw words, cap overflow

        signals = []

        if f1 >= 0.5 and qual < 0.55:
            signals.append("correct answers with composite quality penalty")
        if f1 < 0.3 and gnd >= 0.5:
            signals.append("wrong but grounded answers — verbosity or format issue")
        if f1 < 0.3 and gnd < 0.4 and faith < 0.4:
            signals.append("all metrics low — possible hallucination or retrieval gap")
        if ctx_r < 0.4 and f1 < 0.3:
            signals.append("retrieval-adjacent failures")
        if ans_r < 0.3 and f1 >= 0.4:
            signals.append("question-answer embedding mismatch — possible metric artifact")
        if length >= 10 and f1 < 0.4:
            signals.append("verbose answers with low correctness")
        if length <= 3 and qual < 0.55:
            signals.append("short answers penalized by composite quality metric")

        if not signals:
            signals.append("borderline metric values across multiple dimensions")

        return "Likely: " + "; ".join(signals)

    def print_cluster_summary(self, inv_cluster_result: Dict) -> None:
        """
        Print cluster analysis results to console.
        Each cluster shows: size, metric profile, variant distribution,
        hypothesis, and centroid example for human review.
        """
        if inv_cluster_result is None:
            print("⚠️  No cluster results to display.")
            return

        inv_clusters = inv_cluster_result['clusters']

        print("\n" + "="*80)
        print("🔬 LAYER 2: UNK CLUSTER ANALYSIS")
        print("="*80)
        print(f"\n  UNK cases clustered: {inv_cluster_result['n_unk']}")
        print(f"  K (clusters):        {inv_cluster_result['k']}")
        print(f"  Silhouette score:    {inv_cluster_result['silhouette']:.3f} "
              f"({'good' if inv_cluster_result['silhouette'] > 0.3 else 'weak — clusters may overlap'})")
        print(f"\n  Purpose: Review each cluster and decide whether it represents")
        print(f"  a new named pattern (promote to rule) or edge cases (dismiss).\n")

        for cid, cluster in sorted(inv_clusters.items()):
            prof = cluster['metric_profile']
            cent = cluster['centroid_item']

            print(f"{'─'*70}")
            print(f"  CLUSTER {cid} — {cluster['size']} cases "
                  f"({cluster['pct_of_unk']:.0%} of UNK)")
            print(f"  Hypothesis: {cluster['hypothesis']}")
            print(f"  Variants: "
                  + ", ".join(f"{v}={n}" for v, n in cluster['variant_dist'].items()))

            print(f"\n  Avg Metric Profile:")
            print(f"    F1={prof['f1_score']:.2f} | "
                  f"Ground={prof['groundedness']:.2f} | "
                  f"Faith={prof['faithfulness']:.2f} | "
                  f"Complete={prof['completeness']:.2f}")
            print(f"    AnsRel={prof['answer_relevancy']:.2f} | "
                  f"CtxRel={prof['context_relevancy']:.2f} | "
                  f"Quality={prof['quality_score']:.2f} | "
                  f"AvgLen={min(prof['answer_length'], 9999):.1f}w")

            print(f"\n  Centroid Example (most representative case):")
            print(f"    Q:      {cent['question'][:80]}...")
            print(f"    Gold:   '{cent['gold_answer']}'")
            print(f"    Answer: '{cent['generated_answer'][:70]}'")
            print(f"    F1={cent['f1_score']:.2f} | "
                  f"Ground={cent.get('groundedness') or 0:.2f} | "
                  f"Faith={cent.get('faithfulness') or 0:.2f} | "
                  f"HitRate={cent['hit_rate']:.0%}")

            print(f"\n  ❓ Human Review Decision:")
            print(f"     [ ] Promote to new rule-based pattern (P16+)")
            print(f"     [ ] Dismiss — edge cases / metric artifacts")
            print(f"     [ ] Keep as 'Emerging Pattern {cid}' for monitoring")
            print()

        print("="*80)
        print("  After review: add new patterns to PATTERN_DEFINITIONS")
        print("  and implement in _diagnose_pattern() to reduce future UNK rate.")
        print("="*80)

    def export_cluster_cases(self,
                              inv_cluster_result: Dict,
                              output_dir: str = 'results/low_quality_analysis'
                              ) -> None:
        """
        Export per-cluster markdown files for human review.
        Each file contains: cluster profile, centroid example,
        all cases in the cluster, and a decision template.
        """
        if inv_cluster_result is None:
            return

        inv_out = Path(output_dir) / 'unk_clusters'
        inv_out.mkdir(parents=True, exist_ok=True)

        for cid, cluster in inv_cluster_result['clusters'].items():
            prof = cluster['metric_profile']
            cent = cluster['centroid_item']

            inv_md = f"""# UNK Cluster {cid} — Human Review

## Cluster Summary

| Field | Value |
|-------|-------|
| Cluster ID | {cid} |
| Size | {cluster['size']} cases ({cluster['pct_of_unk']:.0%} of all UNK) |
| Silhouette Quality | {inv_cluster_result['silhouette']:.3f} |
| Hypothesis | {cluster['hypothesis']} |
| Variant Distribution | {', '.join(f"{v}={n}" for v, n in cluster['variant_dist'].items())} |

## Average Metric Profile

| Metric | Avg Value | vs Threshold |
|--------|-----------|-------------|
| F1 Score | {prof['f1_score']:.3f} | {'✅' if prof['f1_score'] >= 0.5 else '⚠️'} (≥0.50) |
| Groundedness | {prof['groundedness']:.3f} | {'✅' if prof['groundedness'] >= 0.5 else '⚠️'} (≥0.50) |
| Faithfulness | {prof['faithfulness']:.3f} | {'✅' if prof['faithfulness'] >= 0.7 else '⚠️'} (≥0.70) |
| Completeness | {prof['completeness']:.3f} | {'✅' if prof['completeness'] >= 0.4 else '⚠️'} (≥0.40) |
| Answer Relevancy | {prof['answer_relevancy']:.3f} | {'✅' if prof['answer_relevancy'] >= 0.5 else '⚠️'} (≥0.50) |
| Context Relevancy | {prof['context_relevancy']:.3f} | {'✅' if prof['context_relevancy'] >= 0.4 else '⚠️'} (≥0.40) |
| Quality Score | {prof['quality_score']:.3f} | {'✅' if prof['quality_score'] >= 0.7 else '⚠️'} (≥0.70) |
| Avg Answer Length | {min(prof['answer_length'], 9999):.1f} words | — |

## Centroid Example (Most Representative Case)

**Question:** {cent['question']}

**Gold Answer:** `{cent['gold_answer']}`

**Generated Answer:** `{cent['generated_answer']}`

| Metric | Value |
|--------|-------|
| F1 Score | {cent['f1_score']:.3f} |
| Groundedness | {cent.get('groundedness') or 0:.3f} |
| Faithfulness | {cent.get('faithfulness') or 0:.3f} |
| Completeness | {cent.get('completeness') or 0:.3f} |
| Hit Rate | {cent['hit_rate']:.1%} |

## All Cases in This Cluster

| # | Q ID | Variant | F1 | Ground | Faith | Quality | Answer |
|---|------|---------|-----|--------|-------|---------|--------|
"""
            for rank, item in enumerate(cluster['items'], 1):
                inv_md += (
                    f"| {rank} | Q{item['question_id']} | {item['variant']} | "
                    f"{item['f1_score']:.2f} | "
                    f"{item.get('groundedness') or 0:.2f} | "
                    f"{item.get('faithfulness') or 0:.2f} | "
                    f"{item.get('quality_score') or 0:.2f} | "
                    f"{item['generated_answer'][:40]}... |\n"
                )

            inv_md += f"""
## Human Review Decision

> **Instructions:** Review the centroid example and the metric profile above.
> Decide whether this cluster represents a coherent, named failure pattern.

- [ ] **Promote to new pattern (P{16 + cid}):** This cluster has a clear, consistent failure mode
  - Proposed pattern name: _______________
  - Proposed rule condition: _______________
  - Proposed mitigations: _______________

- [ ] **Dismiss — edge cases / metric artifacts:** No coherent pattern, likely noise

- [ ] **Keep as Emerging Pattern {cid}:** Pattern exists but needs more data to define rules

---
*Cluster {cid} | {cluster['size']} cases | Silhouette: {inv_cluster_result['silhouette']:.3f}*
*Generated by LLM Evaluation Framework — Layer 2 Cluster Analysis*
"""
            inv_md_file = inv_out / f"cluster_{cid}_review.md"
            with open(inv_md_file, 'w', encoding='utf-8') as f:
                f.write(inv_md)

        print(f"✓ Cluster review files exported to: {inv_out}/")
        print(f"  Files: " +
              ", ".join(f"cluster_{cid}_review.md"
                        for cid in inv_cluster_result['clusters']))


# =============================================================================