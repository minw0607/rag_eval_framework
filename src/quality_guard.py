"""
QualityGuard — real-time quality gate for every generated RAG answer.

Design
------
Runs before the answer is returned to the caller. Uses embedding similarity
(no LLM call, no external API required in "local" mode) to catch the most
common failure modes:

  Groundedness  — is the answer supported by the retrieved context?
                  Method: MiniMax avg_i(max_j(sim(answer_sent_i, ctx_sent_j)))
                  Low score (< 0.5) → likely hallucination

  Completeness  — does the answer cover the retrieved context adequately?
                  Method: inverse MiniMax avg_j(max_i(sim(ctx_sent_j, ans_sent_i)))
                  Low score (< 0.4) → answer is too sparse / misses key info

  Answer relevancy — is the answer on-topic for the question?
                     Method: cosine_sim(answer_embedding, question_embedding)

  Context relevancy — is the retrieved context on-topic for the question?
                      Method: cosine_sim(context_embedding, question_embedding)

  Faithfulness (embedding tier 1) — avg claim-level support score
                                    (RAGAS LLM judge is called separately via EvalMetrics)

  Conciseness — penalises very long answers relative to question length
  Relevance SNR — ratio of on-topic tokens to total tokens (simple heuristic)
  Quality score — weighted composite of the above (see Config.QUALITY_WEIGHTS)

Limitations
-----------
- Embedding-based groundedness / completeness can produce false positives when
  the answer and context share surface-level vocabulary without factual overlap.
- "local" mode (sentence-transformers) is ~10% weaker than API embeddings.
- Length checks use character counts as a proxy for token counts.
"""

import re
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    if _HAS_SKLEARN:
        return float(sk_cosine(a.reshape(1, -1), b.reshape(1, -1))[0][0])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (M x N) for M row-vectors in A and N in B."""
    if _HAS_SKLEARN:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sk_cosine(A, B)
    norms_a = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    norms_b = np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    return (A / norms_a) @ (B / norms_b).T


def _split_sentences(text: str, min_len: int = 8) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= min_len] or [text]


class QualityGuard:
    """
    Real-time embedding-based quality gate for RAG answers.

    Initialise once per session, then call validate() for each answer.

    Parameters
    ----------
    embedding_client : openai.OpenAI or sentence_transformers.SentenceTransformer
        If None, initialises a local SentenceTransformer model automatically.
    embedding_model : str
        Model name / deployment name passed to the embedding_client.
    config : Config, optional
        Framework Config class. Uses default thresholds/weights if not provided.
    use_local : bool
        Force local SentenceTransformer even if embedding_client is provided.
    """

    def __init__(self,
                 embedding_client=None,
                 embedding_model: str = "all-mpnet-base-v2",
                 config=None,
                 use_local: bool = False):
        self.embedding_model  = embedding_model
        self.config           = config
        self.use_local        = use_local or (embedding_client is None)
        self.embedding_client = embedding_client

        if self.use_local:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embedding mode. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            self._local_model = None

        # Resolve thresholds and weights from config or defaults
        if config is not None:
            self.thresholds = config.THRESHOLDS
            self.weights    = config.QUALITY_WEIGHTS
        else:
            self.thresholds = {
                "f1_correct":        0.5,
                "groundedness":      0.5,
                "faithfulness":      0.7,
                "answer_relevancy":  0.5,
                "context_relevancy": 0.4,
            }
            self.weights = {
                "f1_score":         0.35,
                "faithfulness":     0.25,
                "conciseness":      0.15,
                "relevance_snr":    0.15,
                "answer_relevancy": 0.10,
            }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def validate(self,
                 answer: str,
                 context_docs: List[Dict],
                 question: str = None,
                 gold_answer: str = None
                 ) -> Tuple[bool, List[str], Dict]:
        """
        Validate a generated answer.

        Returns
        -------
        is_valid : bool
        issues   : List[str]  — human-readable failure descriptions
        metrics  : Dict       — all computed metric values
        """
        metrics = self._compute_metrics(answer, context_docs, question)
        issues  = self._check_thresholds(metrics, answer)
        is_valid = len(issues) == 0
        return is_valid, issues, metrics

    def get_info(self) -> Dict:
        return {
            "embedding_model": self.embedding_model,
            "provider":        "local (sentence-transformers)" if self.use_local else "api",
            "thresholds":      self.thresholds,
            "weights":         self.weights,
        }

    # =========================================================================
    # METRIC COMPUTATION
    # =========================================================================

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, D) ndarray."""
        if self.use_local:
            return self._local_model.encode(texts, normalize_embeddings=True)
        response = self.embedding_client.embeddings.create(
            input=texts, model=self.embedding_model
        )
        return np.array([item.embedding for item in response.data])

    def _compute_metrics(self, answer: str, context_docs: List[Dict],
                         question: str = None) -> Dict:
        metrics = {
            "groundedness":      0.0,
            "completeness":      0.0,
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_relevancy": 0.0,
            "conciseness":       1.0,
            "relevance_snr":     0.5,
            "quality_score":     0.0,
        }

        if not answer or not context_docs:
            return metrics

        try:
            # Build text pools
            ans_sentences = _split_sentences(answer)
            ctx_sentences = []
            full_context  = ""
            for doc in context_docs:
                content = doc.get("content", "")
                ctx_sentences.extend(_split_sentences(content))
                full_context += " " + content
            full_context = full_context.strip()

            if not ctx_sentences:
                return metrics

            # Embed everything in as few calls as possible
            texts_to_embed = ans_sentences + ctx_sentences
            has_question = bool(question)
            if has_question:
                texts_to_embed = [question] + texts_to_embed

            embs = self._embed(texts_to_embed)

            offset = 0
            q_emb = None
            if has_question:
                q_emb  = embs[0]
                offset = 1

            ans_embs = embs[offset: offset + len(ans_sentences)]
            ctx_embs = embs[offset + len(ans_sentences):]

            # --- Groundedness: avg_i max_j sim(ans_i, ctx_j) ---
            sim_matrix = _cosine_matrix(ans_embs, ctx_embs)
            groundedness = float(np.mean(np.max(sim_matrix, axis=1)))

            # --- Completeness (inverse MiniMax): avg_j max_i sim(ctx_j, ans_i) ---
            completeness = float(np.mean(np.max(sim_matrix.T, axis=1)))

            # --- Faithfulness (embedding tier 1, same as groundedness here) ---
            faithfulness = groundedness

            # --- Answer relevancy ---
            answer_relevancy = 0.5
            if q_emb is not None:
                # Mean of answer sentence similarities to question
                ans_q_sims = _cosine_matrix(ans_embs, q_emb.reshape(1, -1)).flatten()
                answer_relevancy = float(np.mean(ans_q_sims))

            # --- Context relevancy ---
            context_relevancy = 0.5
            if q_emb is not None:
                ctx_q_sims = _cosine_matrix(ctx_embs, q_emb.reshape(1, -1)).flatten()
                context_relevancy = float(np.mean(ctx_q_sims))

            # --- Conciseness (penalise answers longer than 3× question length) ---
            conciseness = 1.0
            if question:
                q_words = max(1, len(question.split()))
                a_words = len(answer.split())
                ratio = a_words / q_words
                if ratio > 3:
                    conciseness = max(0.0, 1.0 - (ratio - 3) / 10)

            # --- Relevance SNR (ratio of answer words appearing in context) ---
            ctx_words_lower = set(full_context.lower().split())
            ans_words_lower = answer.lower().split()
            if ans_words_lower:
                snr_hits = sum(1 for w in ans_words_lower if w in ctx_words_lower)
                relevance_snr = snr_hits / len(ans_words_lower)
            else:
                relevance_snr = 0.0

            # --- Quality score (weighted composite) ---
            quality_score = (
                self.weights.get("faithfulness",     0.25) * faithfulness +
                self.weights.get("conciseness",      0.15) * conciseness +
                self.weights.get("relevance_snr",    0.15) * relevance_snr +
                self.weights.get("answer_relevancy", 0.10) * answer_relevancy
            )
            # f1_score omitted here — computed externally; scale the remaining weights
            quality_score = min(1.0, quality_score / 0.65)

            metrics.update({
                "groundedness":      round(groundedness,      4),
                "completeness":      round(completeness,      4),
                "faithfulness":      round(faithfulness,      4),
                "answer_relevancy":  round(answer_relevancy,  4),
                "context_relevancy": round(context_relevancy, 4),
                "conciseness":       round(conciseness,       4),
                "relevance_snr":     round(relevance_snr,     4),
                "quality_score":     round(quality_score,     4),
            })

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    def _check_thresholds(self, metrics: Dict, answer: str) -> List[str]:
        issues = []

        if len(answer) < 10:
            issues.append("Answer too short (< 10 chars)")
        if len(answer) > 2000:
            issues.append("Answer too long (> 2000 chars)")

        g = metrics.get("groundedness", 0)
        if g < self.thresholds.get("groundedness", 0.5):
            issues.append(f"Low groundedness ({g:.2f} < {self.thresholds['groundedness']}) — possible hallucination")

        c = metrics.get("completeness", 0)
        if c < self.thresholds.get("faithfulness", 0.4):
            issues.append(f"Low completeness ({c:.2f}) — answer may miss key context")

        ar = metrics.get("answer_relevancy", 0)
        if ar < self.thresholds.get("answer_relevancy", 0.5):
            issues.append(f"Low answer relevancy ({ar:.2f}) — answer may be off-topic")

        cr = metrics.get("context_relevancy", 0)
        if cr < self.thresholds.get("context_relevancy", 0.4):
            issues.append(f"Low context relevancy ({cr:.2f}) — retrieval may be poor")

        return issues
