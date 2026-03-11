"""FAISS-based RAG vector database with sentence-transformer embeddings."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RAGDatabase:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # cosine sim after L2 norm
        self.texts: list[str] = []
        self._next_id = 0
        self.id_to_text: dict[int, str] = {}  # ID -> text mapping
        logger.info(f"RAG database initialized (dim={self.dimension})")

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def add(self, text: str) -> int:
        """Add text to RAG. Returns the chunk ID."""
        if not text or not text.strip():
            return -1
        embedding = self._encode([text])
        self.index.add(embedding)
        self.texts.append(text)
        chunk_id = self._next_id
        self.id_to_text[chunk_id] = text
        self._next_id += 1
        logger.info(f"RAG: Added chunk id={chunk_id} (total={len(self.texts)}): {text[:100]}...")
        return chunk_id

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """Query RAG, returns list of texts."""
        if len(self.texts) == 0:
            return []
        k = min(top_k, len(self.texts))
        query_emb = self._encode([query_text])
        scores, indices = self.index.search(query_emb, k)
        results = [self.texts[idx] for idx in indices[0] if idx < len(self.texts)]
        return results

    def query_with_ids(self, query_text: str, top_k: int = 3) -> list[tuple[int, str, float]]:
        """Query RAG, returns list of (id, text, score) tuples."""
        if len(self.texts) == 0:
            return []
        k = min(top_k, len(self.texts))
        query_emb = self._encode([query_text])
        scores, indices = self.index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((int(idx), self.texts[idx], float(score)))
        return results

    @property
    def size(self):
        return len(self.texts)
