"""
Configuration for the RAG evaluation framework.

All secrets are read from environment variables (never hardcoded).
Copy .env.example → .env and fill in your provider details.

Provider auto-detection
-----------------------
Set OPENAI_API_VERSION to activate Azure OpenAI mode.
Leave it blank for OpenAI (direct) or any other OpenAI-compatible endpoint.

  Provider          OPENAI_API_VERSION   Client used
  ──────────────── ─────────────────── ────────────────
  OpenAI (direct)  (blank)              openai.OpenAI
  Azure OpenAI     2025-04-01-preview   openai.AzureOpenAI
  Ollama (local)   (blank)              openai.OpenAI
  Together AI      (blank)              openai.OpenAI
  Groq             (blank)              openai.OpenAI
  LM Studio        (blank)              openai.OpenAI

Call Config.create_client() to get the correctly configured client.
See docs/provider-setup.md for step-by-step setup for each provider.
"""

import os
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Framework-wide configuration. Override any value via environment variables."""

    # =========================================================================
    # LLM PROVIDER
    # =========================================================================
    API_KEY      = os.environ.get("OPENAI_API_KEY", "")
    BASE_URL     = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # Azure OpenAI only — leave blank for all other providers
    API_VERSION  = os.environ.get("OPENAI_API_VERSION", "")

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    GENERATION_MODEL = os.environ.get("OPENAI_GENERATION_MODEL", "gpt-4o")
    TEMPERATURE      = 0.2    # Lower = more deterministic answers (range 0.0–2.0)
    MAX_TOKENS       = 1000

    # =========================================================================
    # EMBEDDING MODEL
    # =========================================================================
    # "large"  → text-embedding-3-large  (3,072 dims — best quality, higher cost)
    # "small"  → text-embedding-3-small  (1,536 dims — good quality, cheaper)
    # "local"  → all-mpnet-base-v2       (768 dims — free, no API key needed)
    EMBEDDING_CHOICE = os.environ.get("EMBEDDING_CHOICE", "small")

    _EMBEDDING_MODELS = {
        "large": {
            "provider": "api",
            "model_name": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "dimensions": 3072,
            "cost_per_1m_tokens": 0.13,
            "typical_tpm_quota": 120_000,
            "description": "Best quality, highest cost",
        },
        "small": {
            "provider": "api",
            "model_name": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "dimensions": 1536,
            "cost_per_1m_tokens": 0.02,
            "typical_tpm_quota": 240_000,
            "description": "Good quality, low cost, 2× faster than large",
        },
        "local": {
            "provider": "local",
            "model_name": "all-mpnet-base-v2",
            "dimensions": 768,
            "cost_per_1m_tokens": 0.0,
            "typical_tpm_quota": float("inf"),
            "description": "Free, no quota limits, ~10% lower quality than API",
        },
    }

    EMBEDDING_CONFIG     = _EMBEDDING_MODELS[EMBEDDING_CHOICE]
    EMBEDDING_MODEL      = EMBEDDING_CONFIG["model_name"]
    EMBEDDING_PROVIDER   = EMBEDDING_CONFIG["provider"]
    EMBEDDING_DIMENSIONS = EMBEDDING_CONFIG["dimensions"]

    # =========================================================================
    # DATA SOURCE
    # =========================================================================
    DATA_SOURCE = "hotpotqa"   # "hotpotqa" | "custom"

    # Path to hotpot_train_v1.1.json — download from hotpotqa.github.io
    DATA_FILE_PATH = Path(os.environ.get("HOTPOTQA_DATA_PATH", "./data/hotpot_train_v1.1.json"))

    # Custom document settings (used when DATA_SOURCE = "custom")
    CUSTOM_DOCS_PATH         = os.environ.get("CUSTOM_DOCS_PATH", "./data/documents")
    CUSTOM_CHUNKING_STRATEGY = "sentence"   # "sentence" | "paragraph" | "sliding_window"
    CUSTOM_CHUNK_SIZE        = 500          # tokens per chunk
    CUSTOM_CHUNK_OVERLAP     = 50

    # =========================================================================
    # DATASET SAMPLING
    # =========================================================================
    USE_SAMPLING     = True
    SAMPLING_METHOD  = "question"    # "question" (recommended) | "document"
    NUM_EVAL_QUESTIONS = 2000
    RANDOM_SEED      = 42

    # =========================================================================
    # CHECKPOINT & CACHE
    # =========================================================================
    FORCE_RESTART            = False   # True → ignore eval checkpoint, start fresh
    FORCE_REBUILD_VECTOR_STORE = False # True → ignore vector store cache, rebuild
    CHECKPOINT_INTERVAL      = 50      # Save eval progress every N questions

    OUTPUT_DIR     = "./outputs"
    CHECKPOINT_DIR = "./checkpoints"

    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    TOP_K                  = 20
    RETRIEVAL_STRATEGY     = "hybrid"   # "semantic" | "keyword" | "hybrid"
    HYBRID_SEMANTIC_WEIGHT = 0.6
    HYBRID_KEYWORD_WEIGHT  = 0.4

    # =========================================================================
    # QUALITY GATES
    # =========================================================================
    MIN_GROUNDEDNESS = 0.5
    MIN_COMPLETENESS = 0.4

    QUALITY_VALIDATION_ENABLED = True
    LOCAL_EMBEDDING_MODEL      = "all-mpnet-base-v2"

    # =========================================================================
    # DIAGNOSTIC TESTS
    # =========================================================================
    RUN_BASELINE_TEST      = True
    RUN_CORRELATION_ANALYSIS = True
    RUN_PROMPT_COMPARISON  = True
    BASELINE_SAMPLE_SIZE   = 100

    # =========================================================================
    # PROMPT VARIANTS  (Multi-Dimensional Framework)
    # =========================================================================
    PROMPT_VARIANTS = {
        # Minimal instruction — establishes a natural performance baseline
        "baseline": (
            "Answer the question using the provided documents.\n"
            "Give a direct, short answer only (1-5 words preferred).\n"
            "Do not explain or add context.\n\n"
            "Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        # Maximum precision, minimal tokens — best for factoid QA and cost-sensitive apps
        "concise": (
            "You are a reading comprehension system. Answer using ONLY the provided documents.\n\n"
            "RULES:\n"
            "1. Use ONLY information explicitly stated in the documents\n"
            "2. Keep answer BRIEF - just the essential facts (typically 1-10 words)\n"
            "3. If the answer cannot be found in the documents, respond: \"Information not available\"\n"
            "4. Do NOT use external knowledge\n"
            "5. Do NOT add extra explanation unless asked\n\n"
            "Documents:\n{context}\n\nQuestion: {question}\n\nBrief answer:"
        ),
        # Comprehensive, self-contained — best for support, education, complex QA
        "detailed": (
            "You are a reading comprehension system. Answer using ONLY the provided documents.\n\n"
            "RULES:\n"
            "1. Use ONLY information explicitly stated in the documents\n"
            "2. Provide a COMPREHENSIVE answer in 2-3 complete sentences\n"
            "3. Include relevant context and details to make the answer self-contained\n"
            "4. If the answer is NOT in the documents, respond: \"Information not available\"\n"
            "5. Do NOT use external knowledge\n\n"
            "Documents:\n{context}\n\nQuestion: {question}\n\nDetailed answer:"
        ),
        # Attribution-enforced — best for legal, medical, financial, regulatory
        "citation": (
            "Answer using ONLY the provided documents. Provide your answer and sources separately.\n\n"
            "OUTPUT FORMAT:\n"
            "Answer: [Your concise answer - typically 1-10 words]\n"
            "Sources: [Document numbers with titles, e.g., Doc 1 (Title A), Doc 3 (Title B)]\n\n"
            "RULES:\n"
            "1. Answer must be BRIEF and DIRECT\n"
            "2. Sources MUST include document number AND title: Doc N (Title)\n"
            "3. If answer is NOT in the documents:\n"
            "   Answer: Information not available\n"
            "   Sources: None\n\n"
            "Documents:\n{context}\n\nQuestion: {question}\n\nOutput:"
        ),
    }

    STYLE_METADATA = {
        "baseline": {
            "description": "Natural, unguided response",
            "primary_benefit": "Establishes performance baseline",
            "use_case": "Benchmarking, understanding default LLM behavior",
            "expected_profile": {"f1": "0.40–0.60", "faithfulness": "0.60–0.70", "length": "10–15 words"},
        },
        "concise": {
            "description": "Precision-optimized, minimal tokens",
            "primary_benefit": "High precision, low cost, fast",
            "primary_tradeoff": "Lower faithfulness scores, limited auditability",
            "use_case": "APIs, cost-sensitive apps, factoid QA",
            "expected_profile": {"f1": "0.70–0.80", "faithfulness": "0.45–0.55", "length": "2–5 words"},
        },
        "detailed": {
            "description": "Completeness-optimized, self-contained",
            "primary_benefit": "High faithfulness, easy to audit",
            "primary_tradeoff": "Higher cost, more verbose",
            "use_case": "Customer support, educational Q&A, complex reasoning",
            "expected_profile": {"f1": "0.60–0.75", "faithfulness": "0.75–0.85", "length": "15–30 words"},
        },
        "citation": {
            "description": "Attribution-enforced, compliance-ready",
            "primary_benefit": "Full traceability, regulatory compliance",
            "primary_tradeoff": "Citation parsing overhead",
            "use_case": "Legal, medical, financial, regulatory domains",
            "expected_profile": {"f1": "0.70–0.80", "faithfulness": "0.50–0.60", "citation_rate": "90–98%"},
        },
    }

    DEFAULT_PROMPT_VARIANT = "concise"

    # =========================================================================
    # QUALITY SCORE WEIGHTS & THRESHOLDS
    # =========================================================================
    THRESHOLDS = {
        "f1_correct":        0.5,
        "groundedness":      0.5,
        "faithfulness":      0.7,
        "answer_relevancy":  0.5,
        "context_relevancy": 0.4,
    }

    QUALITY_WEIGHTS = {
        "f1_score":        0.35,
        "faithfulness":    0.25,
        "conciseness":     0.15,
        "relevance_snr":   0.15,
        "answer_relevancy": 0.10,
    }

    # =========================================================================
    # REPORTING
    # =========================================================================
    GENERATE_EXECUTIVE_REPORT = True
    EXPORT_TO_GITHUB_FORMAT   = True
    INCLUDE_VISUALIZATIONS    = True

    # =========================================================================
    # CLIENT FACTORY
    # =========================================================================

    @classmethod
    def create_client(cls):
        """
        Return the correctly configured LLM client for the active provider.

        Auto-detection rule:
          OPENAI_API_VERSION is set  →  Azure OpenAI  (openai.AzureOpenAI)
          OPENAI_API_VERSION is blank →  OpenAI / any compatible endpoint  (openai.OpenAI)

        Azure OpenAI:
          BASE_URL  = https://<resource-name>.openai.azure.com   (resource endpoint only)
          API_KEY   = key from Azure Portal → Keys and Endpoint
          API_VERSION = e.g. 2025-04-01-preview

        OpenAI (direct) / Ollama / Groq / Together / LM Studio:
          BASE_URL  = provider's base URL (e.g. https://api.openai.com/v1)
          API_KEY   = provider's API key
          API_VERSION = (leave blank)
        """
        if cls.API_VERSION:
            # Azure OpenAI — uses dedicated client that handles api-version and deployment routing
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=cls.API_KEY,
                azure_endpoint=cls.BASE_URL,
                api_version=cls.API_VERSION,
            )
        else:
            # OpenAI (direct) or any OpenAI-compatible endpoint
            from openai import OpenAI
            return OpenAI(
                api_key=cls.API_KEY,
                base_url=cls.BASE_URL,
            )

    # =========================================================================
    # UTILITIES
    # =========================================================================
    @classmethod
    def estimate_embedding_time(cls, num_documents: int) -> dict:
        """Estimate embedding time and cost for a given document count."""
        cfg = cls._EMBEDDING_MODELS[cls.EMBEDDING_CHOICE]
        avg_tokens_per_doc = 200
        total_tokens = num_documents * avg_tokens_per_doc

        if cfg["provider"] == "local":
            docs_per_second = 50
            total_seconds = num_documents / docs_per_second
            return {
                "model": cfg["model_name"],
                "provider": "Local (CPU)",
                "total_time_formatted": cls._format_time(total_seconds),
                "cost_usd": 0.0,
                "quota_limited": False,
                "bottleneck": "CPU speed",
            }

        tpm_quota = cfg["typical_tpm_quota"]
        docs_before_quota = (tpm_quota * 6.5) / avg_tokens_per_doc
        cycles_needed = num_documents / docs_before_quota
        total_minutes = cycles_needed * (6.5 + 55.5)
        cost = (total_tokens / 1_000_000) * cfg["cost_per_1m_tokens"]
        return {
            "model": cfg["model_name"],
            "provider": "API",
            "total_time_formatted": cls._format_time(total_minutes * 60),
            "cost_usd": cost,
            "quota_limited": True,
            "cycles_needed": int(cycles_needed),
            "bottleneck": f"TPM quota ({tpm_quota:,})",
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
