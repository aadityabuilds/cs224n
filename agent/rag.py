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
        logger.info(f"RAG database initialized (dim={self.dimension})")

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def add(self, text: str):
        if not text or not text.strip():
            return
        embedding = self._encode([text])
        self.index.add(embedding)
        self.texts.append(text)
        logger.info(f"RAG: Added chunk (total={len(self.texts)}): {text[:100]}...")

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        if len(self.texts) == 0:
            return []
        k = min(top_k, len(self.texts))
        query_emb = self._encode([query_text])
        scores, indices = self.index.search(query_emb, k)
        results = [self.texts[idx] for idx in indices[0] if idx < len(self.texts)]
        return results

    @property
    def size(self):
        return len(self.texts)
