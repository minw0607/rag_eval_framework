"""
EnhancedRAGSystem — memory-optimized RAG generation with quality gates.

Design decisions
----------------
Temperature 0.2 (not 1.0):
    HotpotQA is a factoid benchmark — deterministic generation produces
    measurably higher F1 and faithfulness scores. Temperature 1.0 was
    the single largest source of score variance in baseline experiments.

Top-K 20 (not 5):
    Multi-hop questions in HotpotQA often require evidence from 2+ documents.
    K=5 caused frequent retrieval misses; K=20 with hybrid re-ranking captures
    both relevant documents reliably.

Document truncation (max 500 tokens):
    Long HotpotQA paragraphs cause context windows to fill with low-signal
    text. Truncating to 500 tokens per document preserves the key facts
    without crowding out evidence from other retrieved documents.

Memory cleanup after each question:
    With 2000+ questions × 4 prompt variants, unreleased references to large
    context strings accumulate quickly. Explicit gc.collect() after each
    question keeps resident memory stable across the full evaluation run.

Prompt delegation:
    This class uses the default prompt from Config.PROMPT_VARIANTS. The full
    multi-prompt comparison (baseline/concise/detailed/citation) is done by
    EvalMetrics.evaluate_multi_prompt(), which calls the LLM directly.

Limitations
-----------
- answer_question() uses the default prompt variant from Config.
  For full multi-prompt evaluation, use EvalMetrics.evaluate_multi_prompt().
- QualityGuard runs embedding-based checks only; RAGAS faithfulness and
  completeness cascade are computed separately in the evaluation loop.
"""

import gc
from typing import List, Dict, Tuple, Optional


class EnhancedRAGSystem:
    """
    RAG system with hybrid retrieval, document truncation, and quality gates.

    Parameters
    ----------
    vector_store      : VectorStore
    generation_client : openai.OpenAI  (any OpenAI-compatible client)
    quality_guard     : QualityGuard
    config            : Config
    """

    def __init__(self, vector_store, generation_client, quality_guard, config):
        self.vector_store       = vector_store
        self.generation_client  = generation_client
        self.quality_guard      = quality_guard
        self.config             = config

        self.max_doc_tokens            = 500    # chars ≈ tokens × 4
        self.max_total_context_tokens  = 4000
        self.last_usage                = None   # stores response.usage for cost tracking

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def answer_question(self, question: str,
                        top_k: int = None,
                        prompt_variant: str = None
                        ) -> Tuple[str, List[Dict], Dict]:
        """
        Retrieve → generate → validate for a single question.

        Returns
        -------
        answer         : str
        docs_summary   : List[Dict]  — lightweight metadata only (title, score, preview)
        quality_metrics: Dict        — groundedness, completeness, etc.
        """
        if top_k is None:
            top_k = self.config.TOP_K
        if prompt_variant is None:
            prompt_variant = getattr(self.config, "DEFAULT_PROMPT_VARIANT", "concise")

        retrieved_docs = self.vector_store.search(
            question, top_k=top_k, strategy=self.config.RETRIEVAL_STRATEGY,
            semantic_weight=self.config.HYBRID_SEMANTIC_WEIGHT,
            keyword_weight=self.config.HYBRID_KEYWORD_WEIGHT
        )

        answer = self._generate_answer(question, retrieved_docs, prompt_variant)

        docs_summary = [
            {
                "title":           doc["metadata"]["title"],
                "doc_id":          doc["metadata"]["doc_id"],
                "score":           doc.get("score", 0.0),
                "content_preview": doc["content"][:200] + "...",
            }
            for doc in retrieved_docs
        ]

        is_valid, issues, metrics = self.quality_guard.validate(
            answer=answer,
            context_docs=retrieved_docs,
            question=question
        )
        metrics["is_valid"] = is_valid
        metrics["issues"]   = issues

        del retrieved_docs
        gc.collect()

        return answer, docs_summary, metrics

    # =========================================================================
    # GENERATION
    # =========================================================================

    def _generate_answer(self, question: str,
                         retrieved_docs: List[Dict],
                         prompt_variant: str) -> str:
        context = self._build_context(retrieved_docs)
        template = self.config.PROMPT_VARIANTS.get(prompt_variant,
                    self.config.PROMPT_VARIANTS.get("concise", ""))
        prompt = template.replace("{context}", context).replace("{question}", question)

        try:
            response = self.generation_client.chat.completions.create(
                model=self.config.GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
            )
            self.last_usage = response.usage
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            answer = "Error generating answer"

        del prompt, context
        gc.collect()
        return answer

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build numbered context string with per-doc and total token limits."""
        parts        = []
        total_tokens = 0
        max_chars    = self.max_doc_tokens * 4

        for i, doc in enumerate(retrieved_docs, 1):
            title   = doc["metadata"]["title"]
            content = doc["content"]
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            est_tokens = len(content) // 4
            if total_tokens + est_tokens > self.max_total_context_tokens:
                break
            parts.append(f"Document {i} ({title}):\n{content}")
            total_tokens += est_tokens

        return "\n\n".join(parts)
