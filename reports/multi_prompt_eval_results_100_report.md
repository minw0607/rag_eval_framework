# RAG Evaluation Report — Comprehensive Validation & Audit Review

**Generated:** 2026-05-24 00:41:27
**Framework:** LLM Evaluation Framework v1.0
**Benchmark:** HotpotQA (Welbl et al., 2018)
**Purpose:** Benchmark evaluation and quality analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Evaluation Scope](#2-evaluation-scope)
3. [Evaluation Approach](#3-evaluation-approach)
4. [Correctness Metrics — Full Results](#4-correctness-metrics--full-results)
5. [Quality Metrics — Full Results](#5-quality-metrics--full-results)
6. [Attribution Metrics — Full Results](#6-attribution-metrics--full-results)
7. [Retrieval Quality Analysis](#7-retrieval-quality-analysis)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Per-Variant Deep Dive](#9-per-variant-deep-dive)
10. [Production Recommendations](#10-production-recommendations)
11. [Cost Analysis](#11-cost-analysis)
12. [Overall Assessment & Conclusion](#12-overall-assessment--conclusion)
13. [Limitations & Known Issues](#13-limitations--known-issues)
14. [Visualization Dashboard](#14-visualization-dashboard)
15. [Files & Reproducibility](#15-files--reproducibility)

---

## 1. Executive Summary

This report presents the results of a comprehensive multi-prompt RAG (Retrieval-Augmented Generation) system evaluation using the LLM Evaluation Framework. **100 questions** from HotpotQA were evaluated across **4 prompt variants** using **13 evaluation metrics** spanning correctness, quality, and attribution dimensions.

### Top-Line Results

| Variant | F1≥0.5 | Faithfulness | Quality Score | Citation Rate | Refusal Rate |
|---------|--------|-------------|--------------|--------------|-------------|
| **Baseline** ⭐ Best F1 | 82.0% | 0.965 | 0.585 | 0.0% | 0.0% |
| **Concise**  | 61.6% | 0.761 | 0.557 | 0.0% | 25.3% |
| **Detailed** 🏆 Best Quality | 0.0% | 0.752 | 0.704 | 12.0% | 27.0% |
| **Citation** 📎 Best Citation | 71.4% | 0.847 | 0.574 | 84.7% | 14.3% |

### Executive Verdict

The Baseline variant performs best, with F1=82.0% and Faith=0.96, demonstrating the highest accuracy and faithfulness, which are critical for reliable information delivery in financial services. The most important risk is the low retrieval hit rate of 37.9%, which undermines evidence coverage and limits answer robustness regardless of generation quality. The system is not ready for production until retrieval performance is significantly improved.


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### Consolidated Findings

**Finding 1 — Retrieval Failures Drive Systemic Weakness** | Severity: 🔴 High  
Persistent underperformance in retrieval is the principal risk, with an average retrieval hit rate of just 37.9% and no questions meeting perfect retrieval criteria (§7). This failure in the RAG backbone undermines groundedness (groundedness <0.50 for 3/4 variants, §5), exacerbates answer hallucination, and diminishes answer correctness (F1 ≥0.5 for Baseline: 82.0%, Concise: 61.6%, and Detailed: 0%) (§4), resulting in elevated production risk.

**Finding 2 — Refusal Rates Compromise Completion and Utility** | Severity: 🔴 High  
Excessive refusal rates in the Concise (25.3%), Detailed (27.0%), and Citation (14.3%) variants exceed the established ≤10% thresholds (§4, §6), directly impairing answerability and utility of the system. These high refusal rates block valid queries even when context is available, impacting user trust and regulatory compliance.

**Finding 3 — Quality, Groundedness, and Answer Relevancy Gaps** | Severity: 🟡 Medium  
Groundedness scores remain below target (Baseline: 0.441, Concise: 0.405, Citation: 0.426; target ≥0.50), correlating with subpar answer and context relevancy (context relevancy avg: 0.384; §5). These issues reflect a failure to reliably link model responses to retrieved evidence, further amplified by weak retrieval, leading to hallucination risk and compounding regulatory gaps.

**Finding 4 — Attribution Shortfall in Citation Responses** | Severity: 🟡 Medium  
The Citation variant demonstrates source attribution at 84.7%, below the ≥90% regulatory target (§6), while refusal in this variant (14.3%) blocks valid responses. These gaps pose risk to transparency and auditability in regulated use cases requiring robust source documentation.

**Finding 5 — High Cascade/Eval Overhead Inflates Cost, Traces to Retrieval Gaps** | Severity: 🟡 Medium  
High cascade LLM trigger rate (79.6% vs. ~20% target, §5) and evaluation framework overhead ($50.74; 69% of total, §11) result in substantial cost inefficiency directly linked to low retrieval quality (§7). The system frequently escalates to secondary evaluation stages, increasing per-question costs due to incomplete, low-confidence initial answers.

**Finding 6 — Inconsistent Completeness and Judge Disagreement** | Severity: 🟡 Medium  
Tier 1/Tier 2 completeness agreement is only 23.1%—well below the target >60% (§5)—indicating significant inconsistencies in completeness judgment between review tiers. This undermines reliability of evaluation outcomes and makes it challenging to certify answer quality at scale.

**Finding 7 — Weak Hit Rate–F1 Correlation Indicates Overreliance on Model Priors** | Severity: 🟡 Medium  
Low correlation between hit rate and F1 (avg: 0.17, §7) suggests the model often answers correctly by relying on prior knowledge rather than retrieved evidence, further heightening explainability and regulatory risks if source-groundedness cannot be demonstrated.

---

**Bottom Line:**  
The RAG system, as assessed, exhibits critical risks in retrieval, grounding, and refusal management that impact correctness, regulatory compliance, and operational cost. Current performance does not meet core SR 26-02 and NIST AI RMF readiness benchmarks, requiring focused remediation on retrieval, attribution, and evaluation workflow to ensure scalable, auditable, and trustworthy deployment.


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### Recommended Next Steps

1. Validate on proprietary post-cutoff data before production deployment
2. Address Detailed prompt verbosity with "state answer first" instruction
3. Schedule quarterly regression testing aligned with vendor model update cycles
4. Implement shadow testing against live traffic before full rollout

---

## 2. Evaluation Scope

### 2.1 System Under Test

The system under test is an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** composed of three layers evaluated as an integrated unit:

```
User Question
     │
     ▼
  Embedding Model ──► Query Vector
     │                    │
     │          ┌─────────▼──────────┐
     │          │   Vector Store     │
     │          │  Hybrid Retrieval  │ ◄── BM25 (keyword, 40%)
     │          │  Dense + Sparse    │ ◄── Embedding (semantic, 60%)
     │          └─────────┬──────────┘
     │                   │  Top-K documents
     ▼                   ▼
  LLM Generation ◄── Context Assembly
     │   (prompt variant applied here)
     ▼
  Raw Answer
     │
     ▼
  QualityGuard ──► Flagged if groundedness < 0.5 or completeness < 0.4
     │
     ▼
  13 Evaluation Metrics (post-hoc, not during generation)
```

**Component inventory for this evaluation run:**

| Component | Configuration | Notes |
|-----------|--------------|-------|
| Embedding model | `text-embedding-3-small` | Used for query encoding and vector store indexing |
| Generation model | `gpt-4o` | LLM receiving retrieved context + prompt |
| Retrieval strategy | hybrid (BM25 40% + dense 60%) | Top-K = 20 documents per query |
| Temperature | 0.2 | Low temperature for reproducibility and faithfulness |
| Prompt variants | 4 (Baseline, Concise, Detailed, Citation) | Each tests a different failure mode |
| QualityGuard gate | Embedding-based pre-filter | Runs before metric aggregation |

### 2.2 Test Data — HotpotQA Benchmark

**HotpotQA** (Yang et al., 2018) is a multi-hop question-answering benchmark requiring reasoning over two or more Wikipedia passages to reach a correct answer. It is the industry-standard benchmark for evaluating retrieval-dependent QA systems because:

- **Multi-hop reasoning** demands co-retrieval of two supporting documents — harder than single-hop retrieval and more representative of real-world knowledge queries.

- **Gold document labels** allow exact measurement of retrieval quality (hit rate), independent of whether the LLM produces a correct answer.

- **Short, precise gold answers** (typically 1–5 words) enable strict token-level F1 scoring alongside reference-free quality metrics.

| Dataset property | Value |
|-----------------|-------|
| Source | HotpotQA training set (hotpotqa.github.io) |
| Total available | ~90,000 multi-hop QA pairs |
| Questions evaluated in this run | **100** |
| Gold supporting documents per question | 2 (required for co-retrieval) |
| Answer format | Short phrase (1–5 words typical) |
| Reasoning type | Bridge and comparison multi-hop |

**Sampling method — question-centric sampling:**  
Questions are sampled first; the document library is then built to guarantee every sampled question's two supporting documents are included in the index. This eliminates retrieval failures due to missing documents, ensuring hit rate reflects retrieval algorithm quality, not index incompleteness.

> ⚠️ **Known limitation:** HotpotQA questions are likely within GPT-4's training data. The expected consequence — weak hit rate ↔ F1 correlation — is observed and documented in §7.3. Validation on proprietary post-cutoff data is recommended before production deployment decisions.

### 2.3 Evaluation Coverage

The evaluation measures system performance across three dimensions:

| Dimension | Metrics Covered | Requires Gold Answer | Primary Use |
|-----------|----------------|---------------------|------------|
| **Correctness** | Exact Match, F1, Contains, ROUGE-L | ✅ Yes | Pass/fail gate, benchmark comparison |
| **Quality** | Groundedness, Completeness, Faithfulness, Answer Relevancy, Context Relevancy, Conciseness, SNR, Quality Score | ❌ No | Production monitoring, hallucination detection |
| **Attribution** | Citation Rate, Refusal Rate | ❌ No | Regulatory compliance, audit trail |

**Prompt variants tested** — each variant is designed to stress-test a different failure mode that may emerge in production:

| Variant | Design Intent | Primary Failure Mode Targeted | Intended Production Use |
|---------|--------------|------------------------------|------------------------|
| **Baseline** | Minimal instruction — tests raw LLM behaviour | Verbosity, lack of focus | General QA, internal tools |
| **Concise** | Strict brevity + explicit refusal instruction | Over-answering, hallucination | APIs, cost-sensitive pipelines |
| **Detailed** | Comprehensive explanation requirement | Under-answering, missing context | Support, education, documentation |
| **Citation** | Mandated Answer + Sources format | Format non-compliance, hallucination | Regulated environments, audit |

*Detailed results for each variant: §4 (Correctness), §5 (Quality), §6 (Attribution), §9 (Per-Variant Deep Dive).*

### 2.4 Regulatory & Compliance Alignment

This evaluation is designed to support model risk management obligations under:

| Framework | Relevance | How This Evaluation Addresses It |
|-----------|----------|----------------------------------|
| **SR 26-02** (Federal Reserve — Model Risk Management) | Requires validation of AI/ML models used in financial services, including documentation, testing, and ongoing monitoring | 13-metric framework with documented thresholds, reproducible results JSON, and audit-grade report |
| **NIST AI RMF** (AI Risk Management Framework) | Structured approach to AI risk: Govern, Map, Measure, Manage | Failure-mode mapping (§2.3), quantitative risk scoring, per-variant risk matrix (§10.2) |

*See §13.3 for a regulatory compliance checklist with current status for each requirement.*

---

## 3. Evaluation Approach

### 3.1 Testing Standards & Design Principles

The framework follows four design principles aligned with SR 26-02 and NIST AI RMF model validation standards:

**1. Reference-free quality measurement.**  
Correctness metrics (F1, Contains) require gold answers and are used for benchmark comparison. Quality metrics (Groundedness, Faithfulness, Completeness, etc.) require only the answer and retrieved context — making them applicable to production monitoring without labelled data.

**2. Two-tier evaluation for accuracy and cost efficiency.**  
Embedding-based metrics are deterministic, fast, and free after the vector store is built. LLM-as-judge metrics are more accurate on edge cases but add API cost. The framework combines both in a cascade: embeddings always run; LLM is invoked only when embedding signals are ambiguous or conflicting.

**3. Multi-prompt comparison.**  
Four prompt variants are evaluated simultaneously on the same question set. This controls for question difficulty and isolates the effect of prompt engineering on accuracy, quality, attribution compliance, and refusal behaviour.

**4. Reproducibility and auditability.**  
All configurations (model names, thresholds, prompt templates, random seed) are fixed and versioned. Results are written to a timestamped JSON file. This report is generated deterministically from that file — the same JSON always produces the same report body (LLM-generated narrative sections are the only non-deterministic elements, and are clearly marked with an AI disclosure).

### 3.2 Evaluation Metrics

**Category 1 — Correctness** *(reference-based; gold answer required)*

| # | Metric | Formula / Method | Threshold | Cost |
|---|--------|-----------------|-----------|------|
| 1 | **Exact Match** | 1 if normalised(prediction) == normalised(gold) else 0 | — | Free |
| 2 | **F1 Score** | 2 × (P × R) / (P + R) on bag-of-words tokens | ≥0.5 per question | Free |
| 3 | **Contains Match** | 1 if gold answer is a substring of prediction | — | Free |
| 4 | **ROUGE-L** | Longest common subsequence F-measure | — | Free |

**Category 2 — Quality** *(reference-free; answer + context only)*

| # | Metric | Formula / Method | Threshold | Cost |
|---|--------|-----------------|-----------|------|
| 5 | **Answer Relevancy** | cosine_sim(embed(answer), embed(question)) | ≥0.50 | Embedding |
| 6 | **Context Relevancy** | cosine_sim(embed(context), embed(question)) | ≥0.40 | Embedding |
| 7 | **Groundedness** | MiniMax: avg_i max_j sim(answer_sent_i, context_sent_j) | ≥0.50 | Embedding |
| 8 | **Faithfulness** | RAGAS: verified_claims / total_claims via LLM entailment | ≥0.70 | LLM |
| 9 | **Completeness** | 2-tier cascade (embedding → conditional LLM) — see §3.4 | ≥0.40 | Embedding + conditional LLM |
| 10 | **Conciseness** | Length penalty relative to question length | ≥0.50 | Free |
| 11 | **Refusal Detection** | Phrase-pattern regex on answer text | ≤10% rate | Free |
| 12 | **Relevance SNR** | Ratio of context-grounded tokens in answer | ≥0.70 | Free |
| 13 | **Quality Score** | Weighted composite of metrics 5–12 | ≥0.70 | — |

**Category 3 — Attribution** *(format compliance)*

| Metric | Method | Threshold | Applies To |
|--------|--------|-----------|-----------|
| **Citation Rate** | Structured parsing for Answer + Sources block | ≥90% | Citation variant |
| **Refusal Rate** | Phrase-pattern detection | ≤10% | All variants |

*Full results by category: §4 (Correctness), §5 (Quality), §6 (Attribution).*

### 3.3 Sampling Method

**Sample size this run:** 100 questions (100/100 evaluated, see §8.2 for per-metric data completeness).

**Question-centric stratified sampling** ensures evaluation integrity:

1. Questions are sampled first from the HotpotQA training set.

2. The document library is built to include all supporting documents for every sampled question — eliminating retrieval failures from missing documents.

3. The random seed is fixed (`RANDOM_SEED` in config) for full reproducibility.

4. All 4 prompt variants are evaluated on the **identical question set** — controlling for question difficulty and isolating prompt effects.

### 3.4 Completeness Cascade — Two-Tier Design

The completeness metric uses a cascade architecture to balance accuracy and cost. The LLM judge (Tier 2) fires on approximately 20% of answers — those where the embedding score is ambiguous or conflicts with other quality signals:

```
Every answer
     │
     ▼
Tier 1 — Embedding (always runs, $0.00 extra per query)
     │  Filter context sentences by cosine similarity to question
     │  Score = avg_i max_j sim(relevant_ctx_sent_i, answer_sent_j)
     │
     ├── Trigger B: |completeness − groundedness| > 0.3
     │              |completeness − faithfulness| > 0.3
     ├── Trigger C: answer < N words AND ≥ M relevant context sentences
     │
     └── No trigger → use Tier 1 score (free)
         Trigger fires (~20% of answers) →
              │
              ▼
         Tier 2 — LLM-as-judge
              Structured prompt: score 0.0–1.0 + one-sentence reason
              Final score = Tier 2 output
```

*Cascade trigger rate and T1/T2 agreement for this run: §11 (Cost Analysis).*

### 3.5 LLM Judge Usage

LLM-as-judge is used in two contexts within the framework, with different scopes and disclosure requirements:

| Usage | Scope | Model | Trigger | Disclosure |
|-------|-------|-------|---------|-----------|
| **Faithfulness (RAGAS)** | Every answer × every variant | `gpt-4o` | Always | Metric result only |
| **Completeness Tier 2** | ~20% of answers (cascade trigger) | `gpt-4o` | Embedding ambiguity | Metric result only |
| **Report narratives** | This report (§1, §4–§11, §12) | `gpt-4o` | Report generation | ⚠️ AI disclosure on every block |

> ⚠️ **Known limitation:** The same model (`{_gen_model}`) serves as both the system under test and the LLM judge. This creates an evaluation dependency that should be documented as a model risk under SR 26-02. See §13.2.

### 3.6 QualityGuard — Real-Time Quality Gate

QualityGuard runs **before** metric aggregation on every answer. Its purpose is to prevent hallucinated or off-topic answers from skewing aggregate statistics — a form of online data quality control:

| Check | Method | Threshold | Action on Failure |
|-------|--------|-----------|------------------|
| Groundedness | MiniMax embedding similarity (answer vs context) | < 0.50 | Answer flagged; metrics still recorded |
| Completeness | Inverse MiniMax (context coverage of answer claims) | < 0.40 | Answer flagged; metrics still recorded |

Flagged answers are included in aggregate metrics (to avoid survivorship bias) but are also written to the low-quality investigation log for pattern analysis.

### 3.7 Output Artifacts & Visualizations

The framework generates three publication-quality figures alongside this report:

| Figure | File | Panels / Contents |
|--------|------|------------------|
| **Evaluation Dashboard** | `evaluation_dashboard.png` | F1 accuracy by variant; Faithfulness distribution; Quality score heatmap; Refusal & citation rates; Cost breakdown (generation vs evaluation); Metric correlation scatter |
| **Retrieval Analysis** | `retrieval_analysis.png` | Hit rate distribution; Hit rate vs F1 correlation by variant; Zero-hit and perfect-hit breakdown |
| **Cost & Cascade Analysis** | `cost_cascade_analysis.png` | Per-question cost by category; Cascade trigger rate; T1/T2 agreement rate; Cost projection curves |

*Figure references: §14 (Visualization Dashboard). All figures are also linked inline at the top of their relevant results sections.*

---

## 4. Correctness Metrics — Full Results

> **Section overview:** Correctness metrics measure whether the RAG system produces answers that match ground-truth references. F1 measures token-level overlap (strict), while Contains tests whether the gold answer appears anywhere in the response (lenient). These are the primary pass/fail signals for factoid question-answering tasks and the foundation for production go/no-go decisions.

> Section 4 tests correctness, measuring the model’s ability to provide accurate answers—a critical prerequisite for reliable production deployment. In this run, the 'Detailed' prompt variant failed severely with 0.0% F1≥0.5 accuracy, and multiple variants had refusal rates above the 10% target (Concise 25.3%, Detailed 27.0%, Citation 14.3%).
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### 4.1 F1 Score — Complete Statistics

| Variant | Mean F1 | Std | Median | Min | Max | F1≥0.5 | Contains |
|---------|---------|-----|--------|-----|-----|--------|---------|
| **Baseline** | 0.790 | 0.377 | 1.000 | 0.000 | 1.000 | 82.0% | 73.0% |
| **Concise** | 0.615 | 0.471 | 1.000 | 0.000 | 1.000 | 61.6% | 61.6% |
| **Detailed** | 0.068 | 0.066 | 0.060 | 0.000 | 0.333 | 0.0% | 65.0% |
| **Citation** | 0.691 | 0.440 | 1.000 | 0.000 | 1.000 | 71.4% | 67.3% |

> **Assessment:** 2/4 variants pass F1≥0.5 Accuracy (70% target) (threshold 0.70). Failing: Concise, Detailed. Note: Detailed variant's near-zero F1 reflects verbosity, not factual incorrectness.

### 4.2 F1 Score Distribution (Binned)

| F1 Range | Baseline | Concise | Detailed | Citation |
|----------|------|------|------|------|
| 0.0–0.3 | 17 (17%) | 37 (37%) | 99 (99%) | 27 (27%) |
| 0.3–0.5 | 1 (1%) | 2 (2%) | 1 (1%) | 3 (3%) |
| 0.5–0.7 | 7 (7%) | 2 (2%) | 0 (0%) | 4 (4%) |
| 0.7–0.9 | 3 (3%) | 1 (1%) | 0 (0%) | 2 (2%) |
| 0.9–1.0 | 72 (72%) | 58 (58%) | 0 (0%) | 64 (64%) |

### 4.3 Refusal Analysis

| Variant | Refusal Rate | Count | Assessment |
|---------|-------------|-------|-----------|
| **Baseline** | 0.0% | ~0/100 | ✅ Never refuses |
| **Concise** | 25.3% | ~25/100 | ⚠️ High — investigate retrieval quality |
| **Detailed** | 27.0% | ~27/100 | ⚠️ High — investigate retrieval quality |
| **Citation** | 14.3% | ~14/100 | 🟡 Borderline — monitor |

### 4.4 Correctness Observations

| # | Observation | Root Cause | Mitigation | Severity |
|---|-------------|-----------|------------|----------|
| 1 | Detailed variant F1≥0.5 accuracy is near-zero (0.0%) despite Contains accuracy of 65.0% | Verbose responses bury the short gold answer; token-overlap F1 systematically penalises long outputs even when the correct fact is present | Add 'state the answer concisely in the first sentence' to the Detailed prompt; use Contains as the primary correctness signal for this variant | 🟡 Medium |
| 2 | F1≥0.5 accuracy below 70% threshold: Concise (61.6%) | Prompt constraints (brevity or citation format) reduce answer precision or increase refusal frequency, lowering token-overlap with gold answers | Run ablation tests isolating each prompt instruction; consider a hybrid instruction that balances brevity with completeness | 🟡 Medium |
| 3 | Concise refusal rate 25.3% exceeds the ≤10% threshold | Refusal instruction triggers when retrieved context is insufficient for multi-hop questions; the model is correctly conservative but over-calibrated | Audit refused questions against their hit_rate; only refuse when hit_rate = 0, not when partial context is available | 🔴 High |
| 4 | Detailed refusal rate 27.0% exceeds the ≤10% threshold | Refusal instruction triggers when retrieved context is insufficient for multi-hop questions; the model is correctly conservative but over-calibrated | Audit refused questions against their hit_rate; only refuse when hit_rate = 0, not when partial context is available | 🔴 High |
| 5 | Citation refusal rate 14.3% exceeds the ≤10% threshold | Refusal instruction triggers when retrieved context is insufficient for multi-hop questions; the model is correctly conservative but over-calibrated | Audit refused questions against their hit_rate; only refuse when hit_rate = 0, not when partial context is available | 🟡 Medium |


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

---

## 5. Quality Metrics — Full Results

> **Section overview:** Quality metrics assess the semantic relationship between answers, retrieved context, and questions — without requiring a gold reference answer. This makes them applicable to production monitoring where labelled data is unavailable. Faithfulness (RAGAS LLM-as-judge) is the primary production signal; Groundedness and Completeness provide complementary coverage. Note that embedding-based metrics (Groundedness, Answer Relevancy) systematically underestimate short answers — this is a known measurement artefact documented in §13.2.

> Section 5 evaluates answer quality, including groundedness, relevancy, and completeness, to ensure outputs are justifiable, relevant, and thorough. Most variants did not meet key thresholds, with groundedness below 0.50 for Baseline (0.441), Concise (0.405), and Citation (0.426), and completeness misalignment rates (Tier 2 triggered 79.6%, Tier 1/2 agreement 23.1%) signaling critical performance gaps.
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### 5.1 All Quality Metrics by Variant

| Metric | Threshold | Baseline | Concise | Detailed | Citation |
|--------|-----------|---------|---------|---------|---------|
| **Groundedness** | ≥0.50 | 0.441 ⚠️ | 0.405 ⚠️ | 0.623 ✅ | 0.426 ⚠️ |
| **Completeness** | ≥0.40 | 0.761 ✅ | 0.642 ✅ | 0.735 ✅ | 0.713 ✅ |
| **Faithfulness** | ≥0.70 | 0.965 ✅ | 0.761 ✅ | 0.752 ✅ | 0.847 ✅ |
| **Answer Relevancy** | ≥0.50 | 0.303 ⚠️ | 0.270 ⚠️ | 0.559 ✅ | 0.289 ⚠️ |
| **Context Relevancy** | ≥0.40 | 0.384 ⚠️ | 0.383 ⚠️ | 0.384 ⚠️ | 0.384 ⚠️ |
| **Conciseness** | ≥0.50 | 1.000 ✅ | 1.000 ✅ | 0.948 ✅ | 1.000 ✅ |
| **Relevance SNR** | ≥0.70 | 0.600 ⚠️ | 0.559 ⚠️ | 0.690 ⚠️ | 0.586 ⚠️ |
| **Quality Score** | ≥0.70 | 0.585 ⚠️ | 0.557 ⚠️ | 0.704 ✅ | 0.574 ⚠️ |

> **Assessment:** All variants pass Faithfulness (threshold 0.70). Best: Baseline (0.96). The system demonstrates consistent faithfulness across all prompt styles.

> **Assessment:** 1/4 variants pass Groundedness (threshold 0.50). Failing: Baseline, Concise, Citation. Note: Groundedness underestimates for short answers.

> **Assessment:** 1/4 variants pass Quality Score (threshold 0.70). Failing: Baseline, Concise, Citation.

> **Answer Relevancy note:** Values below 0.50 for short answers reflect an embedding scaling limitation, not incorrect answers. F1 and Faithfulness are the primary quality signals for short-answer variants.

### 5.2 Quality Observations

| # | Observation | Root Cause | Mitigation | Severity |
|---|-------------|-----------|------------|----------|
| 1 | Groundedness below threshold (≥0.50) for 3/4 variants: Baseline=0.441, Concise=0.405, Citation=0.426 | Short answers (1–5 words) yield low cosine similarity against the full context window — a known embedding-scale artefact, not a hallucination signal | Use Faithfulness (RAGAS) as the primary grounding signal for short-answer variants; consider re-calibrating Groundedness threshold to ≥0.35 for factoid benchmarks | 🟡 Medium |
| 2 | Answer Relevancy below 0.50 for 3/4 variants | Embedding similarity between short factoid answers and full questions is structurally low, independent of answer correctness | Treat Answer Relevancy as informational only for short-answer tasks; do not use as a pass/fail gate in production monitoring | 🟢 Low |
| 3 | Context Relevancy below threshold (≥0.40) for 4/4 variants (avg 0.384) | Retrieved passages contain relevant facts but also significant off-topic content; top-K retrieval returns noise alongside signal | Reduce top-K to 10–15 and re-rank by relevance score; consider query-focused passage extraction before context assembly | 🟡 Medium |
| 4 | Completeness Tier 2 LLM trigger rate 79.6% is well above the ~20% target | Tier 1 embedding thresholds are too narrow, treating most scores as ambiguous and escalating unnecessarily to the more expensive LLM judge | Widen the Tier 1 acceptance band (e.g., completeness > 0.75 → skip Tier 2); analyse which question types drive most escalations | 🟡 Medium |
| 5 | Tier 1 / Tier 2 completeness agreement rate 23.1% is below the >60% target | Embedding and LLM judge frequently disagree, suggesting the embedding completeness metric is poorly calibrated for this task type | Collect a labelled completeness sample; use it to re-calibrate Tier 1 thresholds or adjust the relative weight of the two tiers | 🟡 Medium |


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

---

## 6. Attribution Metrics — Full Results

> **Section overview:** Attribution metrics assess whether the system produces verifiable, source-linked responses — a key requirement for regulated environments under SR 26-02 and NIST AI RMF. Citation Rate measures format compliance with the Answer + Sources structure; Refusal Rate indicates how often the model withholds answers when retrieved context is insufficient, a controlled safety behaviour that must be balanced against user experience.

> Section 6 assesses attribution performance, verifying if answers provide traceable citations to the context, which is essential for transparency and regulatory compliance. The Citation variant achieved only 84.7% attribution—below the required 90%—and exhibited a high refusal rate of 14.3%, indicating both traceability and usability concerns.
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### 6.1 Citation & Format Compliance

| Variant | Citation Rate | Count | Assessment |
|---------|-------------|-------|-----------|
| **Baseline** | 0.0% | ~0/100 | — Not required by prompt |
| **Concise** | 0.0% | ~0/100 | — Not required by prompt |
| **Detailed** | 12.0% | ~12/100 | 📎 Moderate — prompt encourages but does not mandate |
| **Citation** | 84.7% | ~84/100 | ⚠️ Below 90% target |

> **Assessment:** The Citation prompt achieves 84.7% source attribution compliance. ⚠️ Below the ≥90% regulatory target.

### 6.2 Attribution Observations

| # | Observation | Root Cause | Mitigation | Severity |
|---|-------------|-----------|------------|----------|
| 1 | Citation variant source attribution 84.7% is below the ≥90% regulatory target | Model occasionally omits the Sources block when the answer is highly confident or the context is ambiguous | Add post-hoc extraction fallback to detect and append sources; strengthen prompt with an explicit format example and negative example | 🟡 Medium |
| 2 | Citation variant refusal rate 14.3% blocks valid queries that have available context | Attribution requirement and refusal instruction interact to increase refusal frequency beyond what retrieval quality alone would warrant | Decouple refusal from citation: only refuse when hit_rate = 0; allow partial-context answers with explicit uncertainty acknowledgement | 🟡 Medium |


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

---

## 7. Retrieval Quality Analysis

> **Section overview:** Retrieval quality determines whether the RAG system can surface the evidence needed to answer each question. For HotpotQA, this requires co-retrieval of two supporting documents — a harder task than single-hop retrieval. Hit rate measures how often gold documents appear in the top-K results; its correlation with answer accuracy reveals whether the system is genuinely context-dependent or relies primarily on the LLM's parametric knowledge.

> Section 7 examines retrieval effectiveness, as successful answer generation in RAG systems depends on retrieving relevant evidence from the corpus. The average retrieval hit rate was only 37.9%, with zero perfect retrievals across 100 questions, representing a major barrier to overall system reliability.
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### 7.1 Hit Rate Statistics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Average Hit Rate | 37.9% | ⚠️ Below 70% target |
| Perfect Retrievals (100%) | 0/100 (0.0%) | Both gold docs retrieved |
| Zero Retrievals (0%) | 2/100 (2.0%) | Complete retrieval failure |
| Weak Retrievals (<50%) | 68/100 (68.0%) | Partial retrieval |
| Top-K | 20 | Documents retrieved per query |
| Strategy | hybrid | Semantic + keyword hybrid |

> **Assessment:** Retrieval Hit Rate is below threshold at 0.38 (threshold 0.70, gap 0.32).

### 7.2 Hit Rate Distribution

| Hit Rate Range | Count | Pct |
|----------------|-------|-----|
| 0%–25% | 34 | 34.0% |
| 25%–50% | 34 | 34.0% |
| 50%–75% | 23 | 23.0% |
| 75%–100% | 9 | 9.0% |
| 100% (Perfect) | 0 | 0.0% |

### 7.3 Hit Rate vs Accuracy Correlation

| Variant | Correlation | Interpretation |
|---------|------------|----------------|
| **Baseline** | 0.218 | Weak — LLM relies on pre-trained knowledge |
| **Concise** | 0.175 | Weak — LLM relies on pre-trained knowledge |
| **Detailed** | 0.072 | Negligible — possible benchmark data leakage |
| **Citation** | 0.223 | Weak — LLM relies on pre-trained knowledge |

> **Assessment:** Weak hit rate ↔ F1 correlation (~0.1) is expected for HotpotQA given its likely inclusion in GPT-4's training data. Validate on domain-specific corpora where correlation is expected to exceed 0.5.

### 7.4 Retrieval Observations

| # | Observation | Root Cause | Mitigation | Severity |
|---|-------------|-----------|------------|----------|
| 1 | Average retrieval hit rate 37.9% is below the 70% threshold; only 0/100 (0.0%) questions achieved perfect retrieval | HotpotQA requires co-retrieval of two supporting documents; hybrid BM25 + semantic search may not co-rank both within top-20 | Increase top-K from 20 to 30–40; tune BM25/semantic blend weights; consider query expansion for multi-hop questions | 🔴 High |
| 2 | Hit rate ↔ F1 correlation is weak (0.17 avg across variants), indicating the model answers correctly without retrieved context | HotpotQA questions are likely within the model's training data; the LLM draws on parametric knowledge rather than retrieved context | Validate on proprietary post-cutoff documents; measure F1 with and without retrieval to quantify true RAG dependency | 🟡 Medium |


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

---

## 8. Statistical Analysis

> **Section overview:** Statistical tests validate whether observed performance differences are genuine or within sampling noise. The one-sample t-test (H₀: mean F1 = 0.5) confirms which variants demonstrably outperform a random baseline. Data completeness checks verify that metric computation succeeded across all questions and variants — gaps here indicate pipeline errors worth investigating.

> Section 8 explores prompt variant costs and their impact on system-level efficiency and resource allocation. The high cascade (escalation) trigger rate of 79.6% across prompts greatly increased overall evaluation and operational costs for this run.
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

### 8.1 One-Sample T-Test (H₀: Mean F1 = 0.5)

| Variant | Mean F1 | t-statistic | p-value | Significant? |
|---------|---------|------------|---------|-------------|
| **Baseline** | 0.790 | 7.648 | 0.0000 | Yes ✅ (p<0.05) |
| **Concise** | 0.615 | 2.421 | 0.0173 | Yes ✅ (p<0.05) |
| **Detailed** | 0.068 | -65.554 | 0.0000 | No ⚠️ |
| **Citation** | 0.691 | 4.316 | 0.0000 | Yes ✅ (p<0.05) |

### 8.2 Metric Coverage (Data Completeness)

| Metric | Baseline | Concise | Detailed | Citation |
|--------|---------|---------|---------|---------|
| f1_score | 100/100 (100%) ✅ | 100/100 (100%) ✅ | 100/100 (100%) ✅ | 100/100 (100%) ✅ |
| groundedness | 100/100 (100%) ✅ | 99/100 (99%) ✅ | 100/100 (100%) ✅ | 98/100 (98%) ⚠️ |
| completeness | 100/100 (100%) ✅ | 99/100 (99%) ✅ | 100/100 (100%) ✅ | 98/100 (98%) ⚠️ |
| faithfulness | 100/100 (100%) ✅ | 99/100 (99%) ✅ | 100/100 (100%) ✅ | 98/100 (98%) ⚠️ |
| answer_relevancy | 100/100 (100%) ✅ | 99/100 (99%) ✅ | 100/100 (100%) ✅ | 98/100 (98%) ⚠️ |
| quality_score | 100/100 (100%) ✅ | 99/100 (99%) ✅ | 100/100 (100%) ✅ | 98/100 (98%) ⚠️ |

---

## 9. Per-Variant Deep Dive

### 9.1 Baseline Variant

**Description:** Simple, direct prompt with minimal instruction.

#### Correctness

| Metric | Value | vs Threshold | Verdict |
|--------|-------|-------------|---------|
| F1≥0.5 Accuracy | 82.0% | ≥70% target | ✅ |
| Contains Accuracy | 73.0% | — | — |
| Mean F1 | 0.790 (σ=0.377) | ≥0.50 | ✅ |
| Refusal Rate | 0.0% | ≤10% | ✅ |

#### Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Groundedness | 0.441 | ≥0.50 | ⚠️ |
| Completeness | 0.761 | ≥0.40 | ✅ |
| Faithfulness | 0.965 | ≥0.70 | ✅ |
| Quality Score | 0.585 | ≥0.70 | ⚠️ |

**Use when:** General-purpose factoid QA, internal tools.
**Avoid when:** Regulated environments (no citation trail), high-stakes decisions.

---

### 9.2 Concise Variant

**Description:** Strict brevity prompt with explicit refusal instruction.

#### Correctness

| Metric | Value | vs Threshold | Verdict |
|--------|-------|-------------|---------|
| F1≥0.5 Accuracy | 61.6% | ≥70% target | ⚠️ |
| Contains Accuracy | 61.6% | — | — |
| Mean F1 | 0.615 (σ=0.471) | ≥0.50 | ✅ |
| Refusal Rate | 25.3% | ≤10% | ⚠️ |

#### Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Groundedness | 0.405 | ≥0.50 | ⚠️ |
| Completeness | 0.642 | ≥0.40 | ✅ |
| Faithfulness | 0.761 | ≥0.70 | ✅ |
| Quality Score | 0.557 | ≥0.70 | ⚠️ |

**Use when:** API integrations, cost-sensitive pipelines, chatbots.
**Avoid when:** Use cases where refusal frustrates users.

---

### 9.3 Detailed Variant

**Description:** Comprehensive explanation prompt.

#### Correctness

| Metric | Value | vs Threshold | Verdict |
|--------|-------|-------------|---------|
| F1≥0.5 Accuracy | 0.0% | ≥70% target | ⚠️ |
| Contains Accuracy | 65.0% | — | — |
| Mean F1 | 0.068 (σ=0.066) | ≥0.50 | ⚠️ |
| Refusal Rate | 27.0% | ≤10% | ⚠️ |

#### Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Groundedness | 0.623 | ≥0.50 | ✅ |
| Completeness | 0.735 | ≥0.40 | ✅ |
| Faithfulness | 0.752 | ≥0.70 | ✅ |
| Quality Score | 0.704 | ≥0.70 | ✅ |

**Use when:** Support documentation, educational applications.
**Avoid when:** Factoid QA (F1 near-zero due to verbosity), latency-sensitive apps.

---

### 9.4 Citation Variant

**Description:** Regulated-environment prompt mandating Answer + Sources format.

#### Correctness

| Metric | Value | vs Threshold | Verdict |
|--------|-------|-------------|---------|
| F1≥0.5 Accuracy | 71.4% | ≥70% target | ✅ |
| Contains Accuracy | 67.3% | — | — |
| Mean F1 | 0.691 (σ=0.440) | ≥0.50 | ✅ |
| Refusal Rate | 14.3% | ≤10% | ⚠️ |

#### Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Groundedness | 0.426 | ≥0.50 | ⚠️ |
| Completeness | 0.713 | ≥0.40 | ✅ |
| Faithfulness | 0.847 | ≥0.70 | ✅ |
| Quality Score | 0.574 | ≥0.70 | ⚠️ |

**Use when:** Financial services, legal, compliance, regulated environments.
**Avoid when:** High-recall use cases where controlled refusal may frustrate users.

---

## 10. Production Recommendations

### 10.1 Decision Framework

| Requirement | Recommended Variant | Reason |
|-------------|---------------------|--------|
| Source attribution / compliance | Citation | Highest citation rate |
| Cost-sensitive API / factoid QA | Concise | Best F1 with low verbosity |
| Comprehensive explanation | Detailed | Best overall quality score |
| General benchmarking baseline | Baseline | Zero prompt engineering |

### 10.2 Production Risk Matrix

| Variant | Primary Risk | Mitigation |
|---------|-------------|-----------|
| **Baseline** | Hallucinated context (0% refusal) | Add faithfulness monitoring |
| **Concise** | Over-refusal frustrating users | Tune refusal threshold |
| **Detailed** | F1 near-zero (verbosity buries answer) | Add "state answer first" instruction |
| **Citation** | Controlled refusal may block valid queries | Monitor refusal rate in production |

---

## 11. Cost Analysis

> **Section overview:** Cost analysis separates production generation costs (what the RAG system itself spends per query in production) from evaluation framework overhead (what continuous monitoring adds on top). This two-layer view directly answers the operational question: how much does it cost to run the system, and how much to measure it? The cascade trigger rate is the primary lever for controlling evaluation cost without sacrificing coverage.

> Section 11 details end-to-end cost analysis for the evaluation framework, highlighting the financial sustainability of the RAG system in production. The evaluation system’s operational overhead ($50.74, or 69% of total cost) significantly exceeded the core production generation cost ($23.28), driven primarily by frequent LLM cascade triggers.
>> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

| Metric | Value |
|--------|-------|
| Total Cost | $74.0238 |
| Questions Evaluated | 100 |
| Variants Tested | 4 |
| Cost per Question (all variants) | $0.7402 |
| Projected: 1,000 Questions | $740.24 |
| Projected: 10,000 Questions | $7402.38 |

**Cost breakdown:**

| Category | Total | Per Question |
|----------|-------|-------------|
| Generation (production cost) | $23.2833 | $0.2328 |
| Evaluation (framework overhead) | $50.7405 | $0.5074 |

**Cascade evaluation statistics:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Tier 2 LLM trigger rate | 79.6% | ~20% | ⚠️ Elevated |
| T1/T2 agreement rate | 23.1% | >60% | ⚠️ Low |

### 11.1 Cost Observations

| # | Observation | Root Cause | Mitigation | Severity |
|---|-------------|-----------|------------|----------|
| 1 | Evaluation framework overhead ($50.7405, 69% of total) exceeds production generation cost ($23.2833) | Elevated cascade trigger rate drives repeated LLM judge calls; embedding evaluations add fixed per-question overhead on top | Tighten cascade thresholds; for high-volume production monitoring use statistical sampling rather than 100% evaluation coverage | 🟡 Medium |
| 2 | Cascade trigger rate 79.6% is 2–4× the ~20% target, directly inflating per-question evaluation cost | Tier 1 thresholds accept too narrow a band, treating most scores as ambiguous and requiring LLM escalation | Widen the high-confidence acceptance band (e.g., Tier 1 > 0.75 → skip Tier 2); re-evaluate trigger logic with a calibration dataset | 🟡 Medium |


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.

---

## 12. Overall Assessment & Conclusion

**Overall Assessment and Conclusion**

1. **Cross-Metric Synthesis**  
The Baseline variant demonstrates the strongest overall performance, achieving the highest F1 score (82.0%), faithfulness (0.96), and competitive quality (0.59), with no refusals. The Citation variant achieves the highest citation accuracy (84.7%) but at a significant cost to overall F1 (71.4%) and introduces a moderate refusal rate (14.3%). Both Concise and Detailed variants underperform, especially with high refusal rates (25.3% and 27.0%, respectively) and lower F1 scores.

2. **Production Readiness**  
Based on current metrics, the Baseline configuration is the most viable candidate for production deployment, as it offers the best balance of accuracy and faithfulness with minimal refusal. However, the retrieval hit rate of 37.9% and zero citation provision in the Baseline indicate gaps in transparency and explainability required for SR 26-02 / NIST AI RMF compliance.

3. **Pre-Production Requirements**  
- Optimize retrieval pipeline to increase the retrieval hit rate above 50%.  
- Integrate reliable citation generation into the Baseline flow to enhance transparency.  
- Recalibrate refusal thresholds to reduce refusals in the Concise and Detailed variants.  
- Conduct targeted fine-tuning to improve information quality (current highest metric is 0.70 in Detailed).

4. **Framework Validation Status**  
The current Baseline implementation partially satisfies SR 26-02 / NIST AI RMF assurance criteria but falls short on explainability and traceability requirements due to absent citation coverage. Additional improvements are needed prior to full production approval.


> ⚠️ **AI-generated assessment** — findings are derived from automated metric analysis and LLM inference. Independent human review is required before use in any regulatory, audit, or production decision context.


---

## 13. Limitations & Known Issues

### 13.1 Benchmark Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| HotpotQA in GPT-4 training data | Weak hit rate ↔ F1 correlation (~0.1) | Validate on proprietary post-cutoff data |
| Short gold answers (1-5 words) | F1 penalizes verbose answers | Use Contains for Detailed variant |
| Multi-hop questions | Harder retrieval | Expected 70-75% hit rate |
| Static benchmark | No distribution shift testing | Shadow testing on live traffic recommended |

### 13.2 Metric Limitations

| Metric | Known Limitation | Severity |
|--------|----------------|---------|
| Groundedness | Embedding MiniMax underestimates for ≤4 word answers | 🟡 Medium |
| Completeness (Tier 1) | Novel metric, no peer-reviewed validation | 🟡 Medium |
| Faithfulness (RAGAS) | Non-deterministic; same model evaluates its own output | 🟡 Medium |
| Cascade T2 judge | Same model as system under test | 🟡 Medium |

### 13.3 SR 26-02 / Regulatory Considerations

| Consideration | Status |
|--------------|--------|
| Model documentation | ✅ Full metric definitions, formulas, thresholds documented |
| Reproducibility | ✅ Fixed configurations, versioned results files |
| Validation independence | ⚠️ LLM judge uses same model — document as known limitation |
| Ongoing monitoring | ⚠️ Shadow testing and quarterly re-validation recommended |

---

## 14. Visualization Dashboard

### 14.1 Main Evaluation Dashboard (`evaluation_dashboard.png`)

![Evaluation Dashboard](../outputs/evaluation_dashboard.png)

### 14.2 Retrieval Quality Analysis (`retrieval_analysis.png`)

![Retrieval Analysis](../outputs/retrieval_analysis.png)

### 14.3 Cost & Cascade Analysis (`cost_cascade_analysis.png`)

![Cost & Cascade Analysis](../outputs/cost_cascade_analysis.png)

---

## 15. Files & Reproducibility

| File | Description |
|------|-------------|
| `outputs/multi_prompt_eval_results_100.json` | Full results (JSON) |
| `outputs/evaluation_dashboard.png` | Figure 1 — 6-panel main dashboard |
| `outputs/retrieval_analysis.png` | Figure 2 — retrieval quality analysis |
| `outputs/cost_cascade_analysis.png` | Figure 3 — cost & cascade statistics |
| `outputs/low_quality_analysis/` | Pattern analysis + cluster review |

### Reproduction Steps

```python
# 1. Install packages (CELL 1)
# 2. Set up credentials & imports (CELL 2)
# 3. Configure settings (CELL 3)
# 4. Initialize clients (CELL 4)
# 5. Load data: load_hotpotqa() + sample_questions() (CELL 5)
# 6. Build vector store: VectorStore.load_or_build() (CELL 6)
# 7. Initialize RAG system: QualityGuard + EnhancedRAGSystem (CELL 7)
# 8. Pre-flight check: run_preflight_check() (CELL 8)
# 9. Run evaluation: run_evaluation() (CELL 9)
# 10. Visualize: generate_dashboard() (CELL 10)
# 11. Report: generate_report() (CELL 11)
```

---

*LLM Evaluation Framework v1.0 | RAG Evaluation | HotpotQA Benchmark*  
*Generated: 2026-05-24 00:41:27*