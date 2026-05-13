"""
VectorStore — embedding-backed document store with hybrid BM25 + dense retrieval.

Key design decisions
---------------------
Adaptive checkpointing (prevents OOM crashes during long embedding runs):
  0–50%  complete → pickle checkpoint every 100 batches  (fast, small)
  50–80% complete → NumPy checkpoint every 200 batches   (memory efficient)
  80–100%complete → NumPy checkpoint every 500 batches   (rare, large files)

Cache validation on load (prevents silent dimension mismatches):
  Checks doc count, model name, and embedding dimensions before accepting a cache.
  A stale cache from a different embedding model is rejected and rebuilt.

Client re-injection after cache load:
  OpenAI client objects cannot be pickled. After loading from disk,
  call vector_store.set_clients(embedding_client) to restore the connection.

Limitations
-----------
- BM25 index is rebuilt in-memory on load (not stored) — fast for typical datasets.
- Very large corpora (>500k docs) may require the local embedding mode or
  chunking the build into multiple sessions using the checkpoint resume feature.
- Hybrid search normalizes scores independently per query; semantic and keyword
  weights are configurable but not auto-tuned.
"""

import os
import gc
import re
import sys
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class VectorStore:
    """
    Document store with dense (embedding) + sparse (BM25) hybrid retrieval.

    Usage
    -----
    # Build or load from cache:
    store = VectorStore.load_or_build(
        documents=docs_list,
        embedding_client=client,
        embedding_model="text-embedding-3-small",
        config=Config,
        vector_store_file="./checkpoints/vs.pkl",
        checkpoint_file="./checkpoints/vs_checkpoint.pkl",
    )

    # After loading from cache, re-inject the client:
    store.set_clients(embedding_client)

    # Retrieve:
    results = store.search("What causes inflation?", top_k=10)
    """

    def __init__(self, embedding_client, embedding_model: str, config=None):
        self.embedding_client = embedding_client
        self.embedding_model  = embedding_model
        self.config           = config
        self.embeddings:     Optional[np.ndarray] = None
        self.documents:      List[str]  = []
        self.metadata:       List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25:           Optional[BM25Okapi] = None

    # =========================================================================
    # FACTORY — load from cache or build fresh
    # =========================================================================

    @classmethod
    def load_or_build(cls,
                      documents: List[Dict],
                      embedding_client,
                      embedding_model: str,
                      config,
                      vector_store_file: str,
                      checkpoint_file: str,
                      force_rebuild: bool = False) -> "VectorStore":
        """
        Load a cached vector store if valid, otherwise build and cache one.

        Cache validation checks:
        - Document count matches
        - Embedding model name matches
        - Embedding dimensions match config
        """
        vs_path = Path(vector_store_file)

        if vs_path.exists() and not force_rebuild:
            try:
                print(f"Found cache: {vector_store_file} — validating...")
                with open(vs_path, "rb") as f:
                    cache = pickle.load(f)

                issues = []
                if len(cache["documents"]) != len(documents):
                    issues.append(f"Doc count mismatch: cache={len(cache['documents']):,}, need={len(documents):,}")
                if cache.get("model", "") != embedding_model:
                    issues.append(f"Model mismatch: cache={cache.get('model')}, need={embedding_model}")
                if config and cache["embeddings"].shape[1] != config.EMBEDDING_DIMENSIONS:
                    issues.append(f"Dimension mismatch: cache={cache['embeddings'].shape[1]}, need={config.EMBEDDING_DIMENSIONS}")

                if not issues:
                    print(f"Cache valid — loading {len(cache['documents']):,} documents")
                    inst = cls(embedding_client, embedding_model, config)
                    inst.embeddings     = cache["embeddings"]
                    inst.documents      = cache["documents"]
                    inst.metadata       = cache["metadata"]
                    inst.tokenized_docs = cache["tokenized_docs"]
                    inst.bm25           = cache["bm25"]
                    return inst

                print(f"Cache rejected: {'; '.join(issues)}")
                print("Building new vector store...")

            except Exception as e:
                print(f"Cache load error: {e} — rebuilding...")

        inst = cls(embedding_client, embedding_model, config)
        inst.add_documents(documents, checkpoint_file)
        inst.save(vector_store_file)

        # Clean up checkpoint after successful save
        for suffix in (".pkl", ".npy", ".meta", ".tmp"):
            p = Path(checkpoint_file).with_suffix(suffix)
            if p.exists():
                p.unlink(missing_ok=True)

        return inst

    # =========================================================================
    # BUILD
    # =========================================================================

    def add_documents(self, documents: List[Dict], checkpoint_file: str = "checkpoint.pkl"):
        """Embed documents and build BM25 index with adaptive checkpointing."""
        texts    = [doc["content"] for doc in documents]
        metadata = [{"doc_id": doc["doc_id"], "title": doc["title"]} for doc in documents]

        self.documents = texts
        self.metadata  = metadata

        print(f"Embedding {len(texts):,} documents with model '{self.embedding_model}'...")
        self.embeddings = self._build_embeddings_with_checkpoint(texts, checkpoint_file)

        print("Building BM25 index...")
        self.tokenized_docs = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("Done.")

    def _build_embeddings_with_checkpoint(self, texts: List[str],
                                           checkpoint_file: str) -> np.ndarray:
        """Batch-embed texts with quota-aware retry and adaptive checkpointing."""
        batch_size    = 50
        max_retries   = 10
        retry_delays  = [15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600]
        cp_path       = Path(checkpoint_file)
        expected_dims = self.config.EMBEDDING_DIMENSIONS if self.config else None

        # Resume from checkpoint if available
        start_batch, all_embeddings = self._load_checkpoint(cp_path, expected_dims)

        total_batches = (len(texts) + batch_size - 1) // batch_size
        t0 = time.time()
        last_cp_batch = start_batch - 1

        for batch_num in range(start_batch, total_batches):
            batch = texts[batch_num * batch_size: (batch_num + 1) * batch_size]

            success = False
            for retry in range(max_retries):
                try:
                    resp = self.embedding_client.embeddings.create(
                        input=batch, model=self.embedding_model
                    )
                    all_embeddings.extend(np.array(item.embedding) for item in resp.data)
                    success = True
                    break

                except Exception as e:
                    err = str(e)
                    is_quota = any(code in err for code in ("403", "429")) or "quota" in err.lower()

                    if is_quota:
                        m = re.search(r"(\d+):(\d+):(\d+)", err)
                        wait = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                                if m else retry_delays[min(retry, len(retry_delays) - 1)])
                        print(f"\nQuota limit at batch {batch_num+1}. Saving checkpoint, waiting {wait}s...")
                        self._save_checkpoint(cp_path, all_embeddings, batch_num - 1)
                        last_cp_batch = batch_num - 1
                        for remaining in range(wait, 0, -60):
                            print(f"  Waiting: {remaining // 60:02d}:{remaining % 60:02d}", end="\r")
                            time.sleep(min(60, remaining))
                        print()
                    else:
                        print(f"Error batch {batch_num+1} (retry {retry+1}): {err[:80]}")
                        if retry < max_retries - 1:
                            time.sleep(5)

            if not success:
                print(f"Batch {batch_num+1} failed after {max_retries} retries — skipping")

            # Progress
            if (batch_num + 1) % 50 == 0 or batch_num + 1 == total_batches:
                pct = (batch_num + 1) / total_batches * 100
                elapsed = (time.time() - t0) / 60
                print(f"  {pct:.1f}% | batch {batch_num+1}/{total_batches} | {elapsed:.1f} min")

            # Adaptive checkpoint interval
            progress = len(all_embeddings) / len(texts)
            interval = 100 if progress < 0.5 else (200 if progress < 0.8 else 500)
            if batch_num - last_cp_batch >= interval:
                self._save_checkpoint(cp_path, all_embeddings, batch_num)
                last_cp_batch = batch_num
                gc.collect()

        # Final checkpoint
        self._save_checkpoint(cp_path, all_embeddings, total_batches - 1)
        print(f"Embedded {len(all_embeddings):,} documents.")
        return np.array(all_embeddings)

    # =========================================================================
    # CHECKPOINT HELPERS
    # =========================================================================

    def _load_checkpoint(self, cp_path: Path, expected_dims: Optional[int]):
        """Return (start_batch, embeddings_list) from checkpoint, or (0, [])."""
        npy  = cp_path.with_suffix(".npy")
        meta = cp_path.with_suffix(".meta")

        if npy.exists() and meta.exists():
            try:
                with open(meta, "rb") as f:
                    m = pickle.load(f)
                if m.get("model") != self.embedding_model:
                    raise ValueError("model mismatch")
                if expected_dims and m.get("shape", (0, 0))[1] != expected_dims:
                    raise ValueError("dimension mismatch")
                arr = np.load(str(npy))
                print(f"Resuming from NumPy checkpoint: {len(arr):,} docs embedded")
                return m["batch_num"] + 1, arr.tolist()
            except Exception as e:
                print(f"Checkpoint invalid ({e}), starting fresh")
                for p in (npy, meta):
                    p.unlink(missing_ok=True)

        if cp_path.exists():
            try:
                with open(cp_path, "rb") as f:
                    data = pickle.load(f)
                if data.get("model") != self.embedding_model:
                    raise ValueError("model mismatch")
                print(f"Resuming from pickle checkpoint: {len(data['embeddings']):,} docs")
                return data["batch_num"] + 1, data["embeddings"]
            except Exception as e:
                print(f"Checkpoint invalid ({e}), starting fresh")
                cp_path.unlink(missing_ok=True)

        return 0, []

    def _save_checkpoint(self, cp_path: Path, embeddings: List, batch_num: int):
        """Save checkpoint; uses NumPy format when >50% complete (memory-efficient)."""
        progress = len(embeddings) / max(len(self.documents), 1)
        if progress > 0.5:
            self._save_npy_checkpoint(cp_path, embeddings, batch_num)
        else:
            self._save_pkl_checkpoint(cp_path, embeddings, batch_num)

    def _save_npy_checkpoint(self, cp_path: Path, embeddings: List, batch_num: int):
        arr      = np.array(embeddings, dtype=np.float32)
        tmp_npy  = str(cp_path.with_suffix(".tmp"))
        final_npy  = str(cp_path.with_suffix(".npy"))
        final_meta = str(cp_path.with_suffix(".meta"))
        tmp_meta = final_meta + ".tmp"

        np.save(tmp_npy, arr)
        actual_tmp = tmp_npy + ".npy"   # numpy appends .npy
        with open(tmp_meta, "wb") as f:
            pickle.dump({"batch_num": batch_num, "model": self.embedding_model,
                         "shape": arr.shape, "timestamp": time.time()}, f, protocol=4)
        os.replace(actual_tmp, final_npy)
        os.replace(tmp_meta, final_meta)
        del arr
        gc.collect()

    def _save_pkl_checkpoint(self, cp_path: Path, embeddings: List, batch_num: int):
        tmp = str(cp_path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"embeddings": embeddings, "batch_num": batch_num,
                         "model": self.embedding_model, "timestamp": time.time()}, f, protocol=4)
        os.replace(tmp, str(cp_path))
        gc.collect()

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, filepath: str):
        """Persist vector store to disk. Clients are NOT saved (cannot pickle)."""
        data = {
            "embeddings":     self.embeddings,
            "documents":      self.documents,
            "metadata":       self.metadata,
            "tokenized_docs": self.tokenized_docs,
            "bm25":           self.bm25,
            "model":          self.embedding_model,
            "timestamp":      time.time(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=4)
        size_mb = Path(filepath).stat().st_size / (1024 ** 2)
        print(f"Saved vector store: {filepath} ({size_mb:.1f} MB, {len(self.documents):,} docs)")

    @classmethod
    def load(cls, filepath: str, embedding_client, embedding_model: str,
             config=None) -> "VectorStore":
        """Load from disk and re-inject the embedding client."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        inst = cls(embedding_client, embedding_model, config)
        inst.embeddings     = data["embeddings"]
        inst.documents      = data["documents"]
        inst.metadata       = data["metadata"]
        inst.tokenized_docs = data["tokenized_docs"]
        inst.bm25           = data["bm25"]
        print(f"Loaded {len(inst.documents):,} docs, dims={inst.embeddings.shape[1]}")
        return inst

    def set_clients(self, embedding_client) -> bool:
        """
        Re-inject the embedding client after loading from disk.
        Returns True if a smoke-test embedding succeeds.
        """
        self.embedding_client = embedding_client
        try:
            resp = self.embedding_client.embeddings.create(
                input=["smoke test"], model=self.embedding_model
            )
            return len(resp.data[0].embedding) > 0
        except Exception as e:
            print(f"Client verification failed: {e}")
            return False

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def search(self, query: str, top_k: int = 10,
               strategy: str = "hybrid",
               semantic_weight: float = 0.6,
               keyword_weight: float = 0.4) -> List[Dict]:
        """
        Retrieve top_k documents for a query.

        strategy : "semantic" | "keyword" | "hybrid"
        """
        if strategy == "semantic":
            return self._semantic_search(query, top_k)
        if strategy == "keyword":
            return self._keyword_search(query, top_k)
        return self._hybrid_search(query, top_k, semantic_weight, keyword_weight)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.embedding_client.embeddings.create(
            input=[query], model=self.embedding_model
        )
        return np.array(resp.data[0].embedding)

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        q_emb = self._embed_query(query)
        sims  = cosine_similarity(q_emb.reshape(1, -1), self.embeddings)[0]
        idxs  = np.argsort(sims)[::-1][:top_k]
        return [{"content": self.documents[i], "metadata": self.metadata[i],
                 "score": float(sims[i])} for i in idxs]

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        scores = self.bm25.get_scores(query.lower().split())
        idxs   = np.argsort(scores)[::-1][:top_k]
        return [{"content": self.documents[i], "metadata": self.metadata[i],
                 "score": float(scores[i])} for i in idxs]

    def _hybrid_search(self, query: str, top_k: int,
                       sem_w: float, kw_w: float) -> List[Dict]:
        q_emb   = self._embed_query(query)
        sem_sc  = cosine_similarity(q_emb.reshape(1, -1), self.embeddings)[0]
        kw_sc   = self.bm25.get_scores(query.lower().split())

        # Min-max normalise independently
        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-10)

        combined = sem_w * _norm(sem_sc) + kw_w * _norm(kw_sc)
        idxs     = np.argsort(combined)[::-1][:top_k]

        return [{"content":        self.documents[i],
                 "metadata":       self.metadata[i],
                 "score":          float(combined[i]),
                 "semantic_score": float(sem_sc[i]),
                 "keyword_score":  float(kw_sc[i])} for i in idxs]
