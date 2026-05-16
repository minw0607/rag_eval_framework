# Provider Setup Guide

This framework works with any OpenAI-compatible LLM endpoint. The provider is
**auto-detected** from your `.env` file — no code changes required.

**Auto-detection rule:**

| `OPENAI_API_VERSION` | Provider assumed | SDK client used |
|---|---|---|
| Set (e.g. `2025-04-01-preview`) | Azure OpenAI | `openai.AzureOpenAI` |
| Blank | OpenAI / any compatible endpoint | `openai.OpenAI` |

---

## Option A — Azure OpenAI

### What you need from the Azure Portal

1. Go to your Azure OpenAI resource → **Keys and Endpoint**
2. Copy **Key 1** (your API key) and the **Endpoint** (e.g. `https://my-resource.openai.azure.com`)
3. Go to **Model deployments** and note the **deployment names** you created for:
   - Your generation model (e.g. a deployment of GPT-4o)
   - Your embedding model (e.g. a deployment of text-embedding-3-small)

> **Important:** Azure deployment names are yours to choose and do not have to match the underlying model name. The framework passes whatever name you configure directly to the API.

### `.env` configuration

```bash
OPENAI_BASE_URL=https://<your-resource-name>.openai.azure.com
OPENAI_API_KEY=<Key 1 from Azure Portal>
OPENAI_API_VERSION=2025-04-01-preview

# Use your deployment names here, not the underlying model names
OPENAI_GENERATION_MODEL=<your-generation-deployment-name>
OPENAI_EMBEDDING_MODEL=<your-embedding-deployment-name>

EMBEDDING_CHOICE=small
HOTPOTQA_DATA_PATH=./data/hotpot_train_v1.1.json
```

### How the Azure client differs

The framework uses `openai.AzureOpenAI` when `OPENAI_API_VERSION` is set.
This client:
- Authenticates with `api-key` header (not `Authorization: Bearer`)
- Routes requests to `<endpoint>/openai/deployments/<deployment-name>/...`
- Requires `api-version` on every request

All of this is handled automatically by the SDK. You only need to provide the
values above.

### Supported API versions

| Version | Status |
|---|---|
| `2025-04-01-preview` | Current recommended |
| `2024-12-01-preview` | Stable |
| `2024-08-01-preview` | Legacy |

Use the latest preview version unless your deployment has a specific requirement.

---

## Option B — OpenAI (direct)

### What you need

1. Create an account at [platform.openai.com](https://platform.openai.com)
2. Go to **API Keys** and create a new key

### `.env` configuration

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...

# Leave blank — not used for OpenAI direct
OPENAI_API_VERSION=

OPENAI_GENERATION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

EMBEDDING_CHOICE=small
HOTPOTQA_DATA_PATH=./data/hotpot_train_v1.1.json
```

### Choosing models

| Use case | Recommended model |
|---|---|
| Generation (high quality) | `gpt-4o` |
| Generation (lower cost) | `gpt-4o-mini` |
| Embedding (best quality) | `text-embedding-3-large` → set `EMBEDDING_CHOICE=large` |
| Embedding (balanced) | `text-embedding-3-small` → set `EMBEDDING_CHOICE=small` |
| Embedding (free, no API) | local → set `EMBEDDING_CHOICE=local` |

---

## Option C — Ollama (local, free)

Ollama lets you run open-source models locally at no cost.

### Setup

```bash
# Install Ollama: https://ollama.com
ollama pull llama3          # or mistral, phi3, etc.
ollama serve                # starts the local API server
```

### `.env` configuration

```bash
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama       # any non-empty string works

OPENAI_API_VERSION=         # leave blank

OPENAI_GENERATION_MODEL=llama3
EMBEDDING_CHOICE=local      # sentence-transformers — no embedding API needed
```

> **Note:** Ollama does not support the OpenAI embeddings endpoint by default.
> Set `EMBEDDING_CHOICE=local` to use sentence-transformers instead.

---

## Option D — Other compatible providers

These providers expose an OpenAI-compatible API, so no code changes are needed.

| Provider | `OPENAI_BASE_URL` | Notes |
|---|---|---|
| **Groq** | `https://api.groq.com/openai/v1` | Fast inference, free tier |
| **Together AI** | `https://api.together.xyz/v1` | Many open models |
| **Anyscale** | `https://api.endpoints.anyscale.com/v1` | |
| **LM Studio** | `http://localhost:1234/v1` | Local GUI, like Ollama |
| **vLLM** | `http://localhost:8000/v1` | Self-hosted |

For all of these, leave `OPENAI_API_VERSION` blank.

---

## Key differences: Azure vs OpenAI direct

| | Azure OpenAI | OpenAI (direct) |
|---|---|---|
| **Authentication** | `api-key` header | `Authorization: Bearer` |
| **Model name** | Your deployment name | Public model ID (`gpt-4o`) |
| **Endpoint** | Resource-level URL | `https://api.openai.com/v1` |
| **API version** | Required | Not used |
| **SDK client** | `AzureOpenAI` | `OpenAI` |
| **Billing** | Azure subscription | OpenAI account credits |
| **Data residency** | Configurable per region | US-based |

Both use the same Python SDK (`openai`) — only the client class and init params differ. The framework handles this automatically via `Config.create_client()`.

---

## Verifying your setup

After filling in `.env`, run **Cell 6** in the notebook. A successful output looks like:

**Azure OpenAI:**
```
✓ Azure OpenAI client initialized
  Endpoint:         https://my-resource.openai.azure.com
  Generation model: my-gpt4o-deployment
  Embedding model:  my-embedding-deployment
  API version:      2025-04-01-preview
```

**OpenAI (direct):**
```
✓ OpenAI-compatible client initialized
  Endpoint:         https://api.openai.com/v1
  Generation model: gpt-4o
  Embedding model:  text-embedding-3-small
```

Then run **Cell 25** (quick client test) to confirm both generation and embeddings respond correctly before starting the full evaluation.

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `AuthenticationError` | Wrong API key | Check `OPENAI_API_KEY` in `.env` |
| `NotFoundError` on Azure | Wrong deployment name | Check `OPENAI_GENERATION_MODEL` matches your Azure deployment |
| `InvalidRequestError: api-version` | Azure URL without version | Set `OPENAI_API_VERSION` |
| `Connection refused` | Ollama not running | Run `ollama serve` |
| `Model not found` on Ollama | Model not pulled | Run `ollama pull <model-name>` |
| Embedding dimension mismatch | Changed `EMBEDDING_CHOICE` after caching | Set `FORCE_REBUILD_VECTOR_STORE=True` in `Config` |
