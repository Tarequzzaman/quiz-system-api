from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def _stable_id(docset_id: str, source: str, i: int) -> str:
    """Create a stable chunk ID for a given docset/job_id + file + chunk index."""
    h = hashlib.sha1(f"{docset_id}|{source}|{i}".encode()).hexdigest()
    return f"{docset_id}-{h[:24]}"


def _chunk_text(text: str, max_chars=1200, overlap=200) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    s = (text or "").strip()
    if not s:
        return []
    chunks, i, n = [], 0, len(s)
    while i < n:
        j = min(i + max_chars, n)
        k = s.rfind("\n", i + 200, j)
        if k == -1:
            k = s.rfind(" ", i + 200, j)
        if k == -1:
            k = j
        chunks.append(s[i:k].strip())
        i = max(k - overlap, i + 1)
    return [c for c in chunks if c]


class RAGIndex:
    def __init__(
        self,
        work_dir: str = "/app/uploaded_files",
        collection: str = None,  # REQUIRED: pass job_id here
    ):
        if not collection:
            raise ValueError("Collection name (e.g., job_id) is required")

        chroma_path = Path(work_dir) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.client = PersistentClient(path=str(chroma_path))

        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBED_MODEL,
        )

        # Create/get a unique collection for this job
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def add_document(self, docset_id: str, source: str, text: str) -> int:
        """Chunk and store text in ChromaDB for this job."""
        chunks = _chunk_text(text)
        if not chunks:
            return 0
        ids = [_stable_id(docset_id, source, i) for i, _ in enumerate(chunks)]
        metas = [
            {"docset_id": docset_id, "source": source, "chunk": i}
            for i, _ in enumerate(chunks)
        ]
        self.collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        return len(chunks)

    def query(self, docset_id: str, question: str, top_k=5) -> List[Dict]:
        """Query the collection for semantically similar chunks."""
        res = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"docset_id": docset_id},
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {
                "text": d,
                "source": m.get("source"),
                "chunk": m.get("chunk"),
                "score": 1 - dist,
            }
            for d, m, dist in zip(docs, metas, dists)
        ]

    def get_all_for_docset(
        self, docset_id: str, limit: Optional[int] = None
    ) -> tuple[list[str], list[dict]]:
        """Fetch all stored documents for a given docset/job_id."""
        res = self.collection.get(
            where={"docset_id": docset_id},
            include=["documents", "metadatas"],
        )
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        if limit:
            docs, metas = docs[:limit], metas[:limit]
        return docs, metas

    def delete_collection(self):
        """Delete this job's collection entirely."""
        self.client.delete_collection(self.collection.name)
