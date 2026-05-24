"""
Cost tracking module for the RAG evaluation framework.

Design philosophy — cost separation
------------------------------------
Every query incurs two distinct cost layers:

  GENERATION COSTS  (production cost of the RAG system itself)
      LLM answer generation = prompt + context + answer tokens

  EVALUATION COSTS  (what the monitoring framework adds on top)
      Embedding   : retrieval query, completeness Tier 1, groundedness check
      LLM judge   : RAGAS faithfulness, completeness Tier 2 cascade

Reporting these separately answers the stakeholder question:
  "Our RAG system costs $X/query in production,
   plus $Y/query to monitor it with this framework."

Limitations
-----------
- Token counts are estimated (char / 4) unless the API returns exact usage.
- Pricing table only covers common OpenAI models; update OPENAI_PRICING for
  other providers or custom deployments.
- Local embeddings (sentence-transformers) are tracked as $0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CostConfig:
    """Pricing configuration (per 1M tokens)."""

    OPENAI_PRICING: Dict = field(default_factory=lambda: {
        # Generation
        "gpt-4o":            {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
        "gpt-4":             {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
        # Embedding
        "text-embedding-3-small": {"embedding": 0.02},
        "text-embedding-3-large": {"embedding": 0.13},
        "text-embedding-ada-002": {"embedding": 0.10},
    })

    LOCAL_EMBEDDING_COST: float = 0.0   # sentence-transformers — free


class CostTracker:
    """
    Track and report RAG system costs with generation vs evaluation separation.

    Provides:
    - Per-query cost breakdown (generation vs evaluation)
    - Run-level totals and summaries
    - Cascade evaluation statistics (trigger rate, agreement rate)
    - Cost scaling projections
    - Pre-run cost estimation

    Usage
    -----
    tracker = CostTracker()
    tracker.track_answer_generation(input_tokens=3000, output_tokens=50, model="gpt-4o")
    tracker.track_completeness_llm(input_tokens=500, output_tokens=80, triggered=True, ...)
    tracker.print_cost_breakdown(num_questions=2000, config_used={...})
    """

    def __init__(self, config: CostConfig = None):
        self.config = config or CostConfig()

        # Generation costs (production)
        self.generation_costs = {"llm_answer_generation": 0.0}
        self.generation_tokens = {
            "prompt_input_tokens":  0,
            "answer_output_tokens": 0,
        }

        # Evaluation costs (framework monitoring)
        self.evaluation_costs = {
            "embedding_retrieval":    0.0,
            "embedding_completeness": 0.0,
            "embedding_groundedness": 0.0,
            "llm_judge_faithfulness": 0.0,
            "llm_judge_completeness": 0.0,
        }
        self.evaluation_tokens = {
            "retrieval_embedding_tokens":     0,
            "completeness_embedding_tokens":  0,
            "groundedness_embedding_tokens":  0,
            "faithfulness_llm_input_tokens":  0,
            "faithfulness_llm_output_tokens": 0,
            "completeness_llm_input_tokens":  0,
            "completeness_llm_output_tokens": 0,
        }

        # Cascade statistics
        self.cascade_stats = {
            "completeness_tier2_triggered": 0,
            "completeness_tier2_agreed":    0,
            "completeness_total_evaluated": 0,
        }
        self._cascade_log: List[Dict] = []

    # =========================================================================
    # GENERATION TRACKING
    # =========================================================================

    def track_answer_generation(self, input_tokens: int, output_tokens: int,
                                 model: str = "gpt-4o") -> float:
        """Track LLM answer generation cost. Called once per question × prompt variant."""
        self.generation_tokens["prompt_input_tokens"]  += input_tokens
        self.generation_tokens["answer_output_tokens"] += output_tokens
        cost = self._llm_cost(input_tokens, output_tokens, model)
        self.generation_costs["llm_answer_generation"] += cost
        return cost

    # =========================================================================
    # EVALUATION TRACKING
    # =========================================================================

    def track_retrieval_embedding(self, query_tokens: int,
                                   model: str = "text-embedding-3-small") -> float:
        self.evaluation_tokens["retrieval_embedding_tokens"] += query_tokens
        cost = self._emb_cost(query_tokens, model)
        self.evaluation_costs["embedding_retrieval"] += cost
        return cost

    def track_completeness_embedding(self, tokens: int,
                                      model: str = "text-embedding-3-small") -> float:
        self.evaluation_tokens["completeness_embedding_tokens"] += tokens
        cost = self._emb_cost(tokens, model)
        self.evaluation_costs["embedding_completeness"] += cost
        return cost

    def track_groundedness_embedding(self, tokens: int,
                                      model: str = "text-embedding-3-small") -> float:
        self.evaluation_tokens["groundedness_embedding_tokens"] += tokens
        cost = self._emb_cost(tokens, model)
        self.evaluation_costs["embedding_groundedness"] += cost
        return cost

    def track_faithfulness_llm(self, input_tokens: int, output_tokens: int,
                                model: str = "gpt-4o") -> float:
        self.evaluation_tokens["faithfulness_llm_input_tokens"]  += input_tokens
        self.evaluation_tokens["faithfulness_llm_output_tokens"] += output_tokens
        cost = self._llm_cost(input_tokens, output_tokens, model)
        self.evaluation_costs["llm_judge_faithfulness"] += cost
        return cost

    def track_completeness_llm(self,
                                input_tokens: int,
                                output_tokens: int,
                                model: str = "gpt-4o",
                                triggered: bool = True,
                                tier1_score: Optional[float] = None,
                                tier2_score: Optional[float] = None,
                                trigger_reasons: Optional[List[str]] = None,
                                agreement: Optional[bool] = None) -> float:
        """Track completeness Tier 2 LLM judge cost and update cascade stats."""
        self.cascade_stats["completeness_total_evaluated"] += 1

        if not triggered:
            return 0.0

        self.cascade_stats["completeness_tier2_triggered"] += 1
        self.evaluation_tokens["completeness_llm_input_tokens"]  += input_tokens
        self.evaluation_tokens["completeness_llm_output_tokens"] += output_tokens

        if agreement is True:
            self.cascade_stats["completeness_tier2_agreed"] += 1

        self._cascade_log.append({
            "tier1_score":     tier1_score,
            "tier2_score":     tier2_score,
            "trigger_reasons": trigger_reasons or [],
            "agreement":       agreement,
        })

        cost = self._llm_cost(input_tokens, output_tokens, model)
        self.evaluation_costs["llm_judge_completeness"] += cost
        return cost

    # =========================================================================
    # BACKWARD-COMPATIBLE ALIASES (keeps existing notebook cells working)
    # =========================================================================

    def track_embedding(self, input_tokens: int,
                        model: str = "text-embedding-3-small") -> float:
        return self.track_retrieval_embedding(query_tokens=input_tokens, model=model)

    def track_query_embedding(self, query_tokens: int,
                               model: str = "text-embedding-3-small") -> float:
        return self.track_retrieval_embedding(query_tokens=query_tokens, model=model)

    def track_llm_generation(self, input_tokens: int, output_tokens: int,
                              model: str = "gpt-4o") -> float:
        return self.track_answer_generation(input_tokens, output_tokens, model)

    def track_quality_validation(self, answer_tokens: int, context_tokens: int,
                                  embedding_model: str = "all-mpnet-base-v2") -> float:
        return self.track_groundedness_embedding(answer_tokens + context_tokens, embedding_model)

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_total_generation_cost(self) -> float:
        return sum(self.generation_costs.values())

    def get_total_evaluation_cost(self) -> float:
        return sum(self.evaluation_costs.values())

    def get_total_cost(self) -> float:
        return self.get_total_generation_cost() + self.get_total_evaluation_cost()

    def get_cascade_trigger_rate(self) -> Optional[float]:
        total = self.cascade_stats["completeness_total_evaluated"]
        return self.cascade_stats["completeness_tier2_triggered"] / total if total else None

    def get_cascade_agreement_rate(self) -> Optional[float]:
        triggered = self.cascade_stats["completeness_tier2_triggered"]
        return self.cascade_stats["completeness_tier2_agreed"] / triggered if triggered else None

    def get_summary(self) -> Dict:
        return {
            "generation_costs":          self.generation_costs,
            "total_generation_cost_usd": self.get_total_generation_cost(),
            "generation_tokens":         self.generation_tokens,
            "evaluation_costs":          self.evaluation_costs,
            "total_evaluation_cost_usd": self.get_total_evaluation_cost(),
            "evaluation_tokens":         self.evaluation_tokens,
            "total_cost_usd":            self.get_total_cost(),
            "cascade_stats":             self.cascade_stats,
            "cascade_trigger_rate":      self.get_cascade_trigger_rate(),
            "cascade_agreement_rate":    self.get_cascade_agreement_rate(),
        }

    def print_cost_breakdown(self, num_questions: int, config_used: Dict) -> Dict:
        gen_model   = config_used.get("generation_model", "gpt-4o")
        emb_model   = config_used.get("embedding_model",  "text-embedding-3-small")
        num_prompts = config_used.get("num_prompts", 4)

        total_gen  = self.get_total_generation_cost()
        total_eval = self.get_total_evaluation_cost()
        total_all  = self.get_total_cost()
        n          = max(num_questions, 1)

        print("=" * 80)
        print("COST BREAKDOWN — Generation vs Evaluation")
        print("=" * 80)
        print(f"\n  Questions evaluated : {num_questions:,}")
        print(f"  Prompt variants     : {num_prompts}")
        print(f"  Generation model    : {gen_model}")
        print(f"  Embedding model     : {emb_model}")

        print(f"\nGENERATION COSTS (RAG system — production):")
        for label, cost in self.generation_costs.items():
            print(f"  {label:<30}  ${cost:.4f}  (${cost/n:.6f}/q)")
        print(f"  {'TOTAL GENERATION':<30}  ${total_gen:.4f}  (${total_gen/n:.6f}/q)")

        print(f"\nEVALUATION COSTS (framework monitoring):")
        for label, cost in self.evaluation_costs.items():
            print(f"  {label:<30}  ${cost:.4f}  (${cost/n:.6f}/q)")
        print(f"  {'TOTAL EVALUATION':<30}  ${total_eval:.4f}  (${total_eval/n:.6f}/q)")

        print(f"\nCOMBINED TOTAL:")
        print(f"  Total cost          : ${total_all:.4f}")
        print(f"  Per question        : ${total_all/n:.6f}")
        eval_pct = (total_eval / total_all * 100) if total_all > 0 else 0
        print(f"  Evaluation overhead : {eval_pct:.1f}% of total")

        trigger_rate   = self.get_cascade_trigger_rate()
        agreement_rate = self.get_cascade_agreement_rate()
        print(f"\nCASCADE STATS (Completeness Tier 2):")
        print(f"  Total evaluated     : {self.cascade_stats['completeness_total_evaluated']:,}")
        triggered_n = self.cascade_stats['completeness_tier2_triggered']
        print(f"  LLM judge triggered : {triggered_n:,}" +
              (f"  ({trigger_rate:.1%})" if trigger_rate else ""))
        agreed_n = self.cascade_stats['completeness_tier2_agreed']
        print(f"  T1/T2 agreement     : {agreed_n:,}" +
              (f"  ({agreement_rate:.1%})" if agreement_rate else ""))

        print(f"\nPROJECTIONS (at current per-query rate):")
        print(f"  {'Scale':>10}  {'Generation':>12}  {'Evaluation':>12}  {'Total':>10}")
        print(f"  {'─' * 50}")
        for scale in [100, 1_000, 10_000, 100_000]:
            g = (total_gen  / n) * scale
            e = (total_eval / n) * scale
            t = g + e
            print(f"  {scale:>10,}q  ${g:>10.2f}  ${e:>10.2f}  ${t:>8.2f}")
        print("=" * 80)
        return self.get_summary()

    def estimate_per_query_cost(self,
                                 avg_query_tokens: int = 15,
                                 avg_context_tokens: int = 3000,
                                 avg_answer_tokens: int = 50,
                                 generation_model: str = "gpt-4o",
                                 embedding_model: str = "text-embedding-3-small",
                                 num_prompts: int = 4,
                                 cascade_trigger_rate: float = 0.20) -> Dict:
        """Estimate per-query cost before running the evaluation."""
        costs: Dict[str, float] = {}

        gen_input  = avg_query_tokens + avg_context_tokens
        gen_output = avg_answer_tokens
        costs["generation_llm"] = self._llm_cost(gen_input, gen_output, generation_model) * num_prompts

        ep = self._emb_rate(embedding_model)
        costs["eval_retrieval_embedding"]    = (avg_query_tokens / 1e6) * ep
        costs["eval_completeness_embedding"] = (500             / 1e6) * ep
        costs["eval_groundedness_embedding"] = ((avg_answer_tokens + avg_context_tokens) / 1e6) * ep

        costs["eval_faithfulness_llm"] = self._llm_cost(
            avg_context_tokens + avg_answer_tokens + 200, 50, generation_model)
        costs["eval_completeness_llm"] = self._llm_cost(
            avg_context_tokens + avg_query_tokens + avg_answer_tokens + 300,
            80, generation_model) * cascade_trigger_rate

        costs["total_generation"] = costs["generation_llm"]
        costs["total_evaluation"] = (
            costs["eval_retrieval_embedding"] +
            costs["eval_completeness_embedding"] +
            costs["eval_groundedness_embedding"] +
            costs["eval_faithfulness_llm"] +
            costs["eval_completeness_llm"]
        )
        costs["total_per_query"] = costs["total_generation"] + costs["total_evaluation"]
        return costs

    def estimate_evaluation_cost(self, num_questions: int, num_prompts: int = 4,
                                  **kwargs) -> Dict:
        pq = self.estimate_per_query_cost(num_prompts=num_prompts, **kwargs)
        return {
            "num_questions":         num_questions,
            "per_query_generation":  pq["total_generation"],
            "per_query_evaluation":  pq["total_evaluation"],
            "per_query_total":       pq["total_per_query"],
            "total_generation_cost": pq["total_generation"] * num_questions,
            "total_evaluation_cost": pq["total_evaluation"] * num_questions,
            "total_cost":            pq["total_per_query"]  * num_questions,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _llm_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = self.config.OPENAI_PRICING.get(model)
        if not pricing:
            # Azure deployment names don't match OpenAI model IDs — fall back to
            # the closest base model by prefix, then default to gpt-4o pricing.
            for base in ("gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5"):
                if base in model:
                    pricing = self.config.OPENAI_PRICING.get(base)
                    break
            if not pricing:
                pricing = self.config.OPENAI_PRICING.get("gpt-4o")  # safe default
        return (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]

    def _emb_cost(self, tokens: int, model: str) -> float:
        pricing = self.config.OPENAI_PRICING.get(model)
        if not pricing:
            return 0.0
        return (tokens / 1e6) * pricing.get("embedding", 0.0)

    def _emb_rate(self, model: str) -> float:
        pricing = self.config.OPENAI_PRICING.get(model, {})
        return pricing.get("embedding", 0.0)
