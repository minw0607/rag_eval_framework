"""
Evaluation metrics library for the RAG evaluation framework.

13 metrics covering lexical, semantic, and LLM-judge evaluation:

  Lexical       : exact_match, contains_match, calculate_f1
  Attribution   : detect_citations, detect_refusal
  Retrieval     : calculate_hit_rate
  Parsing       : parse_structured_response
  Batch         : evaluate_batch, evaluate_multi_prompt
  Faithfulness  : calculate_faithfulness_ragas  (RAGAS LLM-as-judge)
  Completeness  : evaluate_completeness_cascade (2-tier cascade — see below)

Completeness Cascade Design
----------------------------
Completeness is the most computationally expensive metric because embedding
alone is unreliable in two scenarios:

  Tier B (cross-metric disagreement):
      |completeness_t1 - groundedness| > 0.3  OR
      |completeness_t1 - faithfulness| > 0.3
      → conflicting signals mean the embedding score may be wrong

  Tier C (length asymmetry):
      short answer (< N words) AND many relevant context sentences
      → short answers over info-rich context are most prone to false positives

When neither trigger fires, only Tier 1 (embedding, cheap) runs.
When a trigger fires, Tier 2 (LLM judge) overrides Tier 1.
Every decision is logged for full auditability.

Limitations
-----------
- Faithfulness and Tier 2 completeness consume LLM tokens; disable if cost is a concern.
- BERTScore is not included directly — use sentence-transformers embeddings via QualityGuard.
- All embedding-based metrics depend on embedding model quality; "local" mode (~768 dims)
  will produce lower scores than API-based models on the same inputs.
"""

import re
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import warnings as _warnings
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity

def cosine_similarity(A, B):
    """Wrapper suppressing sklearn numerical warnings on near-unit-norm API embeddings."""
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return _sk_cosine_similarity(A, B)


class EvalMetrics:
    """
    Centralized evaluation metrics for RAG system evaluation.

    All metrics follow consistent conventions:
    - Inputs are lowercased and normalized where applicable
    - Returns are floats in [0, 1] range
    - Handles edge cases (empty strings, None values)
    """

    # =========================================================================
    # TEXT NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, remove punctuation, strip articles, collapse whitespace."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text

    # =========================================================================
    # CORE METRICS
    # =========================================================================

    @staticmethod
    def exact_match(prediction: str, gold: str) -> float:
        pred_norm = EvalMetrics.normalize_text(prediction)
        gold_norm = EvalMetrics.normalize_text(gold)
        return 1.0 if pred_norm == gold_norm else 0.0

    @staticmethod
    def contains_match(prediction: str, gold: str) -> float:
        pred_norm = EvalMetrics.normalize_text(prediction)
        gold_norm = EvalMetrics.normalize_text(gold)
        return 1.0 if gold_norm in pred_norm else 0.0

    @staticmethod
    def calculate_f1(prediction: str, gold: str,
                     threshold: float = 0.5) -> Tuple[float, bool]:
        """
        Token-level F1 with normalization and partial match fallback.
        Returns: (f1_score, is_correct)
        """
        pred_norm = EvalMetrics.normalize_text(prediction)
        gold_norm = EvalMetrics.normalize_text(gold)

        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0, False

        pred_counts = Counter(pred_tokens)
        gold_counts = Counter(gold_tokens)
        common = pred_counts & gold_counts
        tp = sum(common.values())

        if tp == 0:
            for gold_tok in gold_tokens:
                for pred_tok in pred_tokens:
                    if len(gold_tok) >= 3 and len(pred_tok) >= 3:
                        if gold_tok in pred_tok or pred_tok in gold_tok:
                            tp += 1
                            break

        precision = tp / len(pred_tokens)
        recall    = tp / len(gold_tokens)

        if precision + recall == 0:
            return 0.0, False

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1, f1 >= threshold

    # =========================================================================
    # CITATION DETECTION
    # =========================================================================

    @staticmethod
    def detect_citations(text: str, retrieved_docs: List[Dict]) -> Tuple[bool, Dict]:
        """Detect citations in formats: [Title], Doc N, Document N (Title)."""
        if not text or not retrieved_docs:
            return False, {'cited_docs': [], 'num_citations': 0, 'cited_titles': []}

        text_lower = text.lower()
        valid_doc_numbers = set(range(1, len(retrieved_docs) + 1))
        doc_titles = {
            i: doc.get('metadata', {}).get('title', f'Document {i}')
            for i, doc in enumerate(retrieved_docs, 1)
        }

        cited_docs, cited_titles = [], []

        for doc_num_str, _ in re.findall(
                r'doc(?:ument)?\s+(\d+)\s*\(([^)]+)\)', text_lower, re.IGNORECASE):
            doc_num = int(doc_num_str)
            if doc_num in valid_doc_numbers:
                cited_docs.append(doc_num)
                cited_titles.append(doc_titles[doc_num])

        for match in re.finditer(r'doc(?:ument)?\s+(\d+)', text_lower):
            doc_num = int(match.group(1))
            if doc_num in valid_doc_numbers:
                cited_docs.append(doc_num)
                cited_titles.append(doc_titles[doc_num])

        for title_match in re.findall(r'\[([^\]]+)\]', text):
            for doc_num, doc_title in doc_titles.items():
                if (title_match.lower() in doc_title.lower() or
                        doc_title.lower() in title_match.lower()):
                    cited_docs.append(doc_num)
                    cited_titles.append(doc_title)
                    break

        cited_docs_unique   = list(dict.fromkeys(cited_docs))
        cited_titles_unique = list(dict.fromkeys(cited_titles))

        return len(cited_docs_unique) > 0, {
            'cited_docs':    cited_docs_unique,
            'cited_titles':  cited_titles_unique,
            'num_citations': len(cited_docs_unique)
        }

    # =========================================================================
    # REFUSAL DETECTION
    # =========================================================================

    _REFUSAL_PHRASES = [
        'information not available', 'cannot answer', 'not found in',
        'unable to answer', 'cannot be determined', 'not provided',
        'not mentioned', 'does not specify', 'not stated',
        'insufficient information'
    ]

    @staticmethod
    def detect_refusal(answer: str) -> bool:
        if not answer:
            return False
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in EvalMetrics._REFUSAL_PHRASES)

    @staticmethod
    def _is_refusal_answer(answer: str) -> bool:
        return EvalMetrics.detect_refusal(answer)

    # =========================================================================
    # RETRIEVAL QUALITY
    # =========================================================================

    @staticmethod
    def calculate_hit_rate(question_item: Dict,
                           retrieved_docs: List[Dict]) -> Tuple[float, Dict]:
        """Hit rate by matching retrieved doc titles against HotpotQA gold titles."""
        gold_titles = set()
        if 'context' in question_item and question_item['context']:
            for title, _ in question_item['context']:
                gold_titles.add(title.strip())

        retrieved_titles = set()
        for doc in retrieved_docs:
            title = doc.get('metadata', {}).get('title', '').strip()
            if title:
                retrieved_titles.add(title)

        if len(gold_titles) == 0:
            return 1.0, {
                'gold_titles': [], 'retrieved_titles': list(retrieved_titles)[:10],
                'found_titles': [], 'missing_titles': []
            }

        found_titles   = gold_titles & retrieved_titles
        missing_titles = gold_titles - retrieved_titles
        hit_rate       = len(found_titles) / len(gold_titles)

        return hit_rate, {
            'gold_titles':      list(gold_titles),
            'retrieved_titles': list(retrieved_titles)[:10],
            'found_titles':     list(found_titles),
            'missing_titles':   list(missing_titles)
        }

    # =========================================================================
    # STRUCTURED RESPONSE PARSING
    # =========================================================================

    @staticmethod
    def parse_structured_response(response_text: str) -> Dict[str, Any]:
        """Parse Answer:/Sources: format produced by the citation prompt variant."""
        lines   = response_text.strip().split('\n')
        answer  = None
        sources = None

        for line in lines:
            line_clean = line.strip()
            if line_clean.lower().startswith('answer:'):
                answer = line_clean[7:].strip()
            elif (line_clean.lower().startswith('sources:') or
                  line_clean.lower().startswith('source:')):
                sources = line_clean.split(':', 1)[1].strip()

        if answer is None:
            answer = response_text.strip()

        return {
            'answer':              answer or response_text.strip(),
            'sources':             sources,
            'parsed_successfully': answer is not None and sources is not None
        }

    # =========================================================================
    # BATCH EVALUATION
    # =========================================================================

    @staticmethod
    def evaluate_batch(predictions: List[str], golds: List[str],
                       threshold: float = 0.5) -> Dict[str, float]:
        if len(predictions) != len(golds):
            raise ValueError("Predictions and golds must have same length")

        exact_matches, contains_matches, f1_scores, correct_f1 = [], [], [], []

        for pred, gold in zip(predictions, golds):
            exact_matches.append(EvalMetrics.exact_match(pred, gold))
            contains_matches.append(EvalMetrics.contains_match(pred, gold))
            f1, is_correct = EvalMetrics.calculate_f1(pred, gold, threshold)
            f1_scores.append(f1)
            correct_f1.append(is_correct)

        return {
            'exact_match': np.mean(exact_matches),
            'contains':    np.mean(contains_matches),
            'avg_f1':      np.mean(f1_scores),
            'f1_accuracy': np.mean(correct_f1),
            'n':           len(predictions)
        }

    @staticmethod
    def evaluate_multi_prompt(question_item: Dict,
                              prompt_variants: Dict,
                              context: str,
                              generation_client,
                              config,
                              retrieved_docs: List[Dict] = None) -> Dict:
        """Evaluate one question across all prompt variants, return per-variant metrics."""
        question = question_item['question']
        gold     = question_item['answer']
        results  = {}

        for variant_name, template in prompt_variants.items():
            try:
                prompt   = template.replace('{context}', context).replace('{question}', question)
                response = generation_client.chat.completions.create(
                    model=config.GENERATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.MAX_TOKENS
                )
                raw_answer   = response.choices[0].message.content.strip()
                parsed       = EvalMetrics.parse_structured_response(raw_answer)
                clean_answer = parsed['answer']
                sources      = parsed['sources']

                has_citation  = False
                citation_info = {'num_citations': 0}
                if retrieved_docs:
                    citation_text = sources if sources else raw_answer
                    has_citation, citation_info = EvalMetrics.detect_citations(
                        citation_text, retrieved_docs)

                f1, correct_f1   = EvalMetrics.calculate_f1(clean_answer, gold)
                correct_contains = EvalMetrics.contains_match(clean_answer, gold)
                is_refusal       = EvalMetrics.detect_refusal(clean_answer)

                results[variant_name] = {
                    'raw_answer':       raw_answer,
                    'clean_answer':     clean_answer,
                    'sources':          sources,
                    'f1_score':         f1,
                    'correct_f1':       correct_f1,
                    'correct_contains': correct_contains,
                    'has_citation':     has_citation,
                    'num_citations':    citation_info.get('num_citations', 0),
                    'is_refusal':       is_refusal,
                    'answer_length':    len(clean_answer.split()),
                    'is_error':         False
                }

            except Exception as e:
                results[variant_name] = {
                    'raw_answer': 'ERROR', 'clean_answer': 'ERROR',
                    'sources': None, 'f1_score': 0.0, 'correct_f1': False,
                    'correct_contains': False, 'has_citation': False,
                    'num_citations': 0, 'is_refusal': False,
                    'answer_length': 0, 'is_error': True, 'error_message': str(e)
                }

        return results

    # =========================================================================
    # FAITHFULNESS — RAGAS LLM-AS-JUDGE
    # =========================================================================

    @staticmethod
    def calculate_faithfulness_ragas(answer: str,
                                     context: str,
                                     generation_client,
                                     model: str,
                                     max_tokens: int = 500) -> Optional[float]:
        """
        RAGAS-style faithfulness: extract atomic claims → verify each against context.
        Returns None for refusals (not penalised), 1.0 if no extractable claims.
        Reference: Es et al. 2023 — https://arxiv.org/abs/2309.15217

        Limitation: 2 LLM calls per answer (extraction + N verification calls).
        Disable via config if cost is a concern.
        """
        if not answer or not context:
            return 0.0
        if EvalMetrics._is_refusal_answer(answer):
            return None

        try:
            extraction_response = generation_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "Extract all atomic factual claims from the following answer. "
                    "An atomic claim is a single, indivisible factual statement.\n\n"
                    "Rules:\n- One claim per line\n- Each claim must be self-contained\n"
                    "- Do not include opinions or hedges\n"
                    "- If the answer has no factual claims, output: NONE\n\n"
                    f"Answer: {answer}\n\nAtomic claims (one per line):"
                )}],
                max_tokens=max_tokens
            )

            claims_text = extraction_response.choices[0].message.content.strip()
            if not claims_text or claims_text.upper() == "NONE":
                return 1.0

            claims = [
                line.strip().lstrip('-•*').strip()
                for line in claims_text.split('\n')
                if line.strip() and line.strip().upper() != 'NONE'
            ]
            if not claims:
                return 1.0

            verified_count = 0
            for claim in claims:
                verification_response = generation_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": (
                        "Does the following context support this claim?\n\n"
                        f"Context:\n{context}\n\nClaim: {claim}\n\n"
                        "Answer YES if the context supports the claim, "
                        "NO if it contradicts or doesn't mention it.\nAnswer (YES/NO):"
                    )}],
                    max_tokens=10
                )
                verdict = verification_response.choices[0].message.content.strip().upper()
                if verdict.startswith('YES'):
                    verified_count += 1

            return verified_count / len(claims)

        except Exception:
            return None

    # =========================================================================
    # COMPLETENESS CASCADE EVALUATION
    # =========================================================================

    @staticmethod
    def _split_sentences(text: str, min_len: int = 10) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if len(s.strip()) >= min_len]

    @staticmethod
    def calculate_completeness_tier1(answer: str,
                                     question: str,
                                     context_docs: List[Dict],
                                     embedding_client,
                                     model: str,
                                     relevance_threshold: float = 0.5,
                                     min_relevant_sentences: int = 1
                                     ) -> Tuple[Optional[float], Dict]:
        """
        Tier 1 — Question-Filtered Context Coverage (embedding-based, always runs).

        Steps:
        1. Extract sentences from all retrieved context docs
        2. Score each sentence by cosine similarity to the question embedding
        3. Keep only sentences above relevance_threshold (filters retrieval noise)
        4. For each relevant sentence, find max cosine similarity to any answer sentence
        5. Mean of those maxima = Tier 1 completeness score

        Returns (score_or_None, metadata_dict).
        """
        meta = {
            'tier':                 1,
            'method':               'question_filtered_context_coverage',
            'n_context_sentences':  0,
            'n_relevant_sentences': 0,
            'relevance_threshold':  relevance_threshold,
        }

        if not answer or not question or not context_docs:
            return 0.0, meta

        if EvalMetrics._is_refusal_answer(answer):
            meta['skipped'] = 'refusal'
            return 0.0, meta

        try:
            ctx_sentences = []
            for doc in context_docs:
                ctx_sentences.extend(EvalMetrics._split_sentences(doc.get('content', '')))

            meta['n_context_sentences'] = len(ctx_sentences)
            if not ctx_sentences:
                return 0.0, meta

            ans_sentences = EvalMetrics._split_sentences(answer) or [answer]

            all_texts = [question] + ctx_sentences + ans_sentences
            response  = embedding_client.embeddings.create(input=all_texts, model=model)
            all_embs  = np.array([item.embedding for item in response.data])

            q_emb    = all_embs[0:1]
            ctx_embs = all_embs[1:1 + len(ctx_sentences)]
            ans_embs = all_embs[1 + len(ctx_sentences):]

            ctx_q_sims   = cosine_similarity(ctx_embs, q_emb).flatten()
            relevant_idx = np.where(ctx_q_sims >= relevance_threshold)[0]

            if len(relevant_idx) < min_relevant_sentences:
                relevant_idx = np.argsort(ctx_q_sims)[-min_relevant_sentences:]

            meta['n_relevant_sentences'] = len(relevant_idx)

            if len(relevant_idx) == 0:
                return 1.0, meta

            relevant_ctx_embs = ctx_embs[relevant_idx]
            sim_matrix        = cosine_similarity(relevant_ctx_embs, ans_embs)
            completeness      = float(np.mean(np.max(sim_matrix, axis=1)))

            return completeness, meta

        except Exception as e:
            meta['error'] = str(e)
            return None, meta

    @staticmethod
    def _should_trigger_completeness_llm(completeness_t1: Optional[float],
                                          groundedness: Optional[float],
                                          faithfulness: Optional[float],
                                          answer: str,
                                          n_relevant_sentences: int,
                                          cross_metric_delta: float = 0.3,
                                          short_answer_words: int = 5,
                                          min_relevant_for_length_check: int = 3
                                          ) -> Tuple[bool, List[str]]:
        """
        Decide whether to trigger Tier 2 (LLM judge). OR logic — any criterion fires.

        Tier B (cross-metric disagreement):
            |completeness_t1 - groundedness| > delta
            |completeness_t1 - faithfulness| > delta

        Tier C (length asymmetry):
            answer < short_answer_words words AND
            n_relevant_sentences >= min_relevant_for_length_check
        """
        reasons = []

        if completeness_t1 is None:
            return False, []

        if groundedness is not None:
            delta_g = abs(completeness_t1 - groundedness)
            if delta_g > cross_metric_delta:
                reasons.append(
                    f"Tier B: |completeness({completeness_t1:.2f}) - "
                    f"groundedness({groundedness:.2f})| = {delta_g:.2f} > {cross_metric_delta}"
                )

        if faithfulness is not None:
            delta_f = abs(completeness_t1 - faithfulness)
            if delta_f > cross_metric_delta:
                reasons.append(
                    f"Tier B: |completeness({completeness_t1:.2f}) - "
                    f"faithfulness({faithfulness:.2f})| = {delta_f:.2f} > {cross_metric_delta}"
                )

        answer_words = len(answer.split()) if answer else 0
        if (answer_words < short_answer_words and
                n_relevant_sentences >= min_relevant_for_length_check):
            reasons.append(
                f"Tier C: short answer ({answer_words} words) with "
                f"{n_relevant_sentences} relevant context sentences — "
                f"embedding completeness unreliable"
            )

        return len(reasons) > 0, reasons

    @staticmethod
    def calculate_completeness_tier2(answer: str,
                                     question: str,
                                     context_docs: List[Dict],
                                     generation_client,
                                     model: str,
                                     max_context_chars: int = 3000
                                     ) -> Tuple[Optional[float], Dict]:
        """
        Tier 2 — LLM-as-judge semantic completeness verification.

        Called only when Tier B or C triggers fire.
        Overrides Tier 1 score when a valid score is returned.
        max_tokens=80 to minimise cost (score + one-sentence reason).
        """
        meta = {'tier': 2, 'method': 'llm_judge', 'model': model}

        if not answer or not question or not context_docs:
            return None, meta

        try:
            t0      = time.time()
            ctx_str = '\n\n'.join([
                f"[Doc {i+1}]: {doc.get('content', '')[:max_context_chars]}"
                for i, doc in enumerate(context_docs)
            ])

            judge_prompt = (
                "You are evaluating whether a generated answer completely covers "
                "all information from the retrieved context that is relevant to the question.\n\n"
                "IMPORTANT: Focus ONLY on context passages relevant to the question. "
                "Ignore context that is not pertinent to the question.\n\n"
                f"Question: {question}\n\n"
                f"Retrieved Context:\n{ctx_str}\n\n"
                f"Generated Answer: {answer}\n\n"
                "Task: Score completeness from 0.0 to 1.0.\n"
                "  1.0 = answer covers all relevant aspects from context\n"
                "  0.5 = answer covers some but misses important relevant info\n"
                "  0.0 = answer misses most relevant information from context\n\n"
                "Respond ONLY in this format:\n"
                "Score: <float between 0.0 and 1.0>\n"
                "Reason: <one sentence>"
            )

            response   = generation_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=80
            )
            latency_ms = int((time.time() - t0) * 1000)
            raw_output = response.choices[0].message.content.strip()

            score  = None
            reason = ''
            for line in raw_output.split('\n'):
                if line.lower().startswith('score:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass
                elif line.lower().startswith('reason:'):
                    reason = line.split(':', 1)[1].strip()

            est_input_tokens  = int(len(judge_prompt.split()) * 1.3)
            est_output_tokens = int(len(raw_output.split()) * 1.3)

            meta.update({
                'latency_ms':        latency_ms,
                'llm_reason':        reason,
                'est_input_tokens':  est_input_tokens,
                'est_output_tokens': est_output_tokens,
                'raw_output':        raw_output,
            })

            return score, meta

        except Exception as e:
            meta['error'] = str(e)
            return None, meta

    @staticmethod
    def evaluate_completeness_cascade(answer: str,
                                       question: str,
                                       context_docs: List[Dict],
                                       embedding_client,
                                       generation_client,
                                       embedding_model: str,
                                       generation_model: str,
                                       groundedness: Optional[float] = None,
                                       faithfulness: Optional[float] = None,
                                       relevance_threshold: float = 0.5,
                                       cross_metric_delta: float = 0.3,
                                       short_answer_words: int = 3,
                                       min_relevant_for_length_check: int = 5
                                       ) -> Dict:
        """
        Full two-tier completeness cascade.

        Returns dict keys:
            completeness          — final score (Tier 2 if triggered, else Tier 1)
            context_coverage      — always Tier 1 score
            tier1_score           — same as context_coverage
            tier2_score           — Tier 2 score or None
            tier2_triggered       — bool
            tier2_trigger_reasons — List[str]
            tier1_meta, tier2_meta, agreement
        """
        result = {
            'completeness':          None,
            'context_coverage':      None,
            'tier1_score':           None,
            'tier2_score':           None,
            'tier2_triggered':       False,
            'tier2_trigger_reasons': [],
            'tier1_meta':            {},
            'tier2_meta':            None,
            'agreement':             None,
        }

        tier1_score, tier1_meta = EvalMetrics.calculate_completeness_tier1(
            answer=answer, question=question, context_docs=context_docs,
            embedding_client=embedding_client, model=embedding_model,
            relevance_threshold=relevance_threshold, min_relevant_sentences=1
        )

        result.update({
            'tier1_score':      tier1_score,
            'context_coverage': tier1_score,
            'completeness':     tier1_score,
            'tier1_meta':       tier1_meta,
        })

        n_relevant = tier1_meta.get('n_relevant_sentences', 0)
        should_trigger, trigger_reasons = EvalMetrics._should_trigger_completeness_llm(
            completeness_t1=tier1_score,
            groundedness=groundedness,
            faithfulness=faithfulness,
            answer=answer,
            n_relevant_sentences=n_relevant,
            cross_metric_delta=cross_metric_delta,
            short_answer_words=short_answer_words,
            min_relevant_for_length_check=min_relevant_for_length_check
        )

        result['tier2_triggered']       = should_trigger
        result['tier2_trigger_reasons'] = trigger_reasons

        if should_trigger and generation_client is not None:
            tier2_score, tier2_meta = EvalMetrics.calculate_completeness_tier2(
                answer=answer, question=question, context_docs=context_docs,
                generation_client=generation_client, model=generation_model
            )
            result['tier2_score'] = tier2_score
            result['tier2_meta']  = tier2_meta

            if tier2_score is not None:
                result['completeness'] = tier2_score
                if tier1_score is not None:
                    result['agreement'] = abs(tier1_score - tier2_score) < 0.35

        return result

    # =========================================================================
    # QUALITY VALIDATION (delegates to QualityGuard)
    # =========================================================================

    @staticmethod
    def validate_quality(answer: str,
                         context_docs: List[Dict],
                         question: str = None,
                         gold_answer: str = None,
                         quality_guard=None,
                         generation_client=None,
                         config=None
                         ) -> Dict:
        """
        Validate answer quality using QualityGuard (optional).
        Returns a dict of quality metrics; all None if quality_guard is not provided.

        Note: returns 'context_coverage' (always Tier 1 embedding score).
        Full cascade completeness is computed separately via evaluate_completeness_cascade().
        """
        empty = {
            'groundedness': None, 'context_coverage': None, 'faithfulness': None,
            'answer_relevancy': None, 'context_relevancy': None,
            'conciseness': None, 'relevance_snr': None,
            'quality_score': None, 'quality_valid': None, 'quality_issues': []
        }

        if quality_guard is None:
            return empty

        try:
            is_valid, issues, metrics = quality_guard.validate(
                answer=answer, context_docs=context_docs,
                question=question, gold_answer=gold_answer
            )

            faithfulness = metrics.get('faithfulness', 0)
            if generation_client is not None and config is not None and answer:
                context_str = '\n\n'.join([doc.get('content', '')[:500] for doc in context_docs])
                ragas_faith = EvalMetrics.calculate_faithfulness_ragas(
                    answer=answer, context=context_str,
                    generation_client=generation_client, model=config.GENERATION_MODEL
                )
                if ragas_faith is not None:
                    faithfulness = ragas_faith

            return {
                'groundedness':      metrics.get('groundedness', 0),
                'context_coverage':  metrics.get('completeness', 0),
                'faithfulness':      faithfulness,
                'answer_relevancy':  metrics.get('answer_relevancy', 0),
                'context_relevancy': metrics.get('context_relevancy', 0),
                'conciseness':       metrics.get('conciseness', None),
                'relevance_snr':     metrics.get('relevance_snr', None),
                'quality_score':     metrics.get('quality_score', 0),
                'quality_valid':     is_valid,
                'quality_issues':    issues
            }

        except Exception as e:
            return {**empty, 'quality_valid': False, 'quality_issues': [f"Validation error: {e}"]}
