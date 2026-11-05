"""Evidence retrieval for EmissioNex.

Primary backend: SentenceTransformer embeddings stored in ChromaDB.
Fallback backend: TF-IDF + cosine similarity, for environments where semantic
retrieval dependencies/model weights are not yet available. The active backend is
reported explicitly so fallback runs are never mistaken for semantic RAG.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_KB = Path("knowledge_base/reference_guides")
DEFAULT_DB = Path("outputs/chroma_db")
EMBEDDING_MODEL = os.getenv("EMISSIONEX_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_chunks(text: str, max_chars: int = 1400, overlap: int = 180) -> list[tuple[str, str]]:
    """Split documents into bounded, overlapping chunks while retaining headings."""
    sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)
    output: list[tuple[str, str]] = []
    for section in sections:
        section = section.strip()
        if len(section) < 40:
            continue
        match = re.search(r"(?m)^#{1,3}\s+(.+)$", section)
        heading = match.group(1).strip() if match else "General guidance"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if len(current) >= 40:
                    output.append((heading, current))
                if len(paragraph) > max_chars:
                    start = 0
                    while start < len(paragraph):
                        end = min(start + max_chars, len(paragraph))
                        part = paragraph[start:end].strip()
                        if len(part) >= 40:
                            output.append((heading, part))
                        if end == len(paragraph):
                            break
                        start = max(end - overlap, start + 1)
                    current = ""
                else:
                    current = paragraph
        if len(current) >= 40:
            output.append((heading, current))
    return output


def read_documents(kb_dir: str | Path = DEFAULT_KB) -> list[dict[str, Any]]:
    """Read supported reference files and create source-traceable chunks."""
    root = Path(kb_dir)
    documents: list[dict[str, Any]] = []
    if not root.exists():
        return documents

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        try:
            pages: list[tuple[int | None, str]] = []
            if suffix in {".md", ".txt"}:
                pages = [(None, path.read_text(encoding="utf-8", errors="ignore"))]
            elif suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                except ImportError:
                    print(f"Skipping PDF (install pypdf): {path}")
                    continue
                reader = PdfReader(str(path))
                pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
            else:
                continue
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue

        for page_no, text in pages:
            for chunk_no, (heading, chunk) in enumerate(_split_chunks(text), start=1):
                source = path.as_posix()
                doc_id = hashlib.sha256(
                    f"{source}|{page_no}|{chunk_no}|{chunk}".encode("utf-8")
                ).hexdigest()[:24]
                documents.append({
                    "id": doc_id,
                    "source": source,
                    "document": path.name,
                    "section": heading,
                    "page": page_no,
                    "chunk_id": chunk_no,
                    "text": chunk,
                })
    return documents


class LocalRAG:
    """RAG retriever with semantic ChromaDB backend and transparent TF-IDF fallback."""

    def __init__(
        self,
        kb_dir: str | Path = DEFAULT_KB,
        db_dir: str | Path = DEFAULT_DB,
        backend: str = "auto",
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.kb_dir = Path(kb_dir)
        self.db_dir = Path(db_dir)
        self.embedding_model_name = embedding_model
        self.documents = read_documents(self.kb_dir)
        self.backend = "empty"
        self.vectorizer = None
        self.matrix = None
        self.collection = None
        self.client = None
        self.embedder = None
        self._doc_by_id = {doc["id"]: doc for doc in self.documents}

        if not self.documents:
            return

        if backend not in {"auto", "semantic", "tfidf"}:
            raise ValueError("backend must be 'auto', 'semantic', or 'tfidf'")
        if backend != "tfidf":
            try:
                self._init_semantic()
                return
            except Exception as exc:
                if backend == "semantic":
                    raise RuntimeError(
                        "Semantic RAG setup failed. Install chromadb and sentence-transformers, "
                        "and ensure the embedding model can be downloaded/cached."
                    ) from exc
                print(f"WARNING: semantic RAG unavailable ({exc}); using TF-IDF fallback.")
        self._init_tfidf()

    def _init_semantic(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        collection_name = "emissionex_reference_chunks"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model_name},
        )
        # Keep batches modest to avoid memory spikes on larger manuals.
        batch_size = 64
        for start in range(0, len(self.documents), batch_size):
            batch = self.documents[start:start + batch_size]
            embeddings = self.embedder.encode(
                [doc["text"] for doc in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            metadatas = []
            for doc in batch:
                metadata = {
                    "source": str(doc["source"]),
                    "document": str(doc["document"]),
                    "section": str(doc["section"]),
                    "page": int(doc["page"] or 0),
                    "chunk_id": int(doc["chunk_id"]),
                }
                metadatas.append(metadata)
            self.collection.add(
                ids=[doc["id"] for doc in batch],
                documents=[doc["text"] for doc in batch],
                metadatas=metadatas,
                embeddings=embeddings,
            )
        self.backend = "semantic_chromadb"

    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000,
        )
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in self.documents])
        self.backend = "tfidf_fallback"

    def search(self, query: str, top_k: int = 4, min_relevance: float = 0.12) -> list[dict[str, Any]]:
        if not query.strip() or not self.documents or top_k <= 0:
            return []
        results: list[dict[str, Any]] = []
        if self.backend == "semantic_chromadb":
            query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
            raw = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, len(self.documents)),
                include=["documents", "metadatas", "distances"],
            )
            ids = (raw.get("ids") or [[]])[0]
            texts = (raw.get("documents") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]
            for doc_id, text, meta, distance in zip(ids, texts, metas, distances):
                score = max(0.0, min(1.0, 1.0 - float(distance)))
                if score < min_relevance:
                    continue
                results.append({
                    "id": doc_id,
                    "source": meta.get("source", "unknown"),
                    "document": meta.get("document", "unknown"),
                    "section": meta.get("section", "General guidance"),
                    "page": meta.get("page") or None,
                    "text": text,
                    "excerpt": _clean_text(text)[:900],
                    "relevance_score": round(score, 4),
                    "retrieval_backend": self.backend,
                })
            return results

        from sklearn.metrics.pairwise import cosine_similarity
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        for idx in scores.argsort()[::-1]:
            score = float(scores[idx])
            if score < min_relevance:
                break
            doc = self.documents[int(idx)]
            results.append({
                **doc,
                "relevance_score": round(score, 4),
                "excerpt": _clean_text(doc["text"])[:900],
                "retrieval_backend": self.backend,
            })
            if len(results) >= top_k:
                break
        return results
