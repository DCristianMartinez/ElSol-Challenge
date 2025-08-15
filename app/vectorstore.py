from __future__ import annotations
from typing import List, Dict, Optional
import math
import re
from collections import Counter, defaultdict

# Qdrant solo se usa en modo embeddings
from qdrant_client import QdrantClient, models

from .openai_utils import get_openai_client
from .config import settings


_token_re = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _token_re.findall(text or "")]


class _KeywordStore:
    """Índice simple en memoria con TF-IDF."""

    def __init__(self) -> None:
        self.docs: Dict[str, Dict] = {}
        self.term_df: Counter[str] = Counter()
        self.doc_tokens: Dict[str, Counter[str]] = {}
        self.N = 0  # número de documentos

    def add(self, doc_id: str, text: str, payload: dict) -> None:
        if doc_id in self.docs:
            prev_terms = self.doc_tokens[doc_id]
            for term in prev_terms:
                self.term_df[term] -= 1
                if self.term_df[term] <= 0:
                    del self.term_df[term]

        tokens = _tokenize(text)
        tf = Counter(tokens)
        self.doc_tokens[doc_id] = tf
        for term in tf.keys():
            self.term_df[term] += 1

        self.docs[doc_id] = {"payload": payload | {"text": text}}
        self.N = len(self.docs)

    def _tfidf_vec(self, tf: Counter[str]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for term, f in tf.items():
            df = self.term_df.get(term, 0)
            if df == 0 or self.N == 0:
                continue
            idf = math.log((1 + self.N) / (1 + df)) + 1.0  # idf suavizado
            vec[term] = (f * idf)
        return vec

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        (small, big) = (a, b) if len(a) < len(b) else (b, a)
        for k, v in small.items():
            if k in big:
                dot += v * big[k]
        for v in a.values():
            na += v * v
        for v in b.values():
            nb += v * v
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / math.sqrt(na * nb)

    def search(self, query: str, top_k: int = 3) -> List:
        q_tf = Counter(_tokenize(query))
        q_vec = self._tfidf_vec(q_tf)
        scores = []
        for doc_id, tf in self.doc_tokens.items():
            d_vec = self._tfidf_vec(tf)
            score = self._cosine(q_vec, d_vec)
            scores.append((score, doc_id))
        scores.sort(reverse=True)
        results = []
        for score, doc_id in scores[:top_k]:
            payload = self.docs[doc_id]["payload"]
            results.append(_ScoredPoint(id=doc_id, score=score, payload=payload))
        return results


class _ScoredPoint:
    """Estructura mínima compatible con lo que usa main.py."""
    def __init__(self, id: str, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = payload


class VectorStore:
    """Usa embeddings de Azure si están configurados; si no, usa _KeywordStore en memoria."""

    def __init__(self, collection: str = "transcripts") -> None:
        self.collection = collection
        self._embeddings_deployment: Optional[str] = settings.azure_openai_embeddings_deployment
        if self._embeddings_deployment:
            # --- MODO EMBEDDINGS (Qdrant en memoria) ---
            self.client = QdrantClient(location=":memory:")
            self._ensure_collection()
            self._mode = "embeddings"
        else:
            # --- MODO KEYWORDS (TF-IDF en memoria) ---
            self._kw = _KeywordStore()
            self._mode = "keywords"

    # -------------- MODO EMBEDDINGS --------------
    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self._embedding_dim(), distance=models.Distance.COSINE),
            )

    def _embedding_dim(self) -> int:
        client = get_openai_client()
        resp = client.embeddings.create(input="ping", model=self._embeddings_deployment)
        return len(resp.data[0].embedding)

    def _embed(self, text: str) -> List[float]:
        client = get_openai_client()
        response = client.embeddings.create(input=text, model=self._embeddings_deployment)
        return response.data[0].embedding

    # -------------- API PÚBLICA --------------
    def add_document(self, doc_id: str, text: str, metadata: dict[str, str]) -> None:
        if self._mode == "embeddings":
            vector = self._embed(text)
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload={"text": text, **metadata},
                    )
                ],
            )
        else:
            self._kw.add(doc_id, text, {"text": text, **metadata})

    def query(self, text: str, top_k: int = 3) -> List:
        if self._mode == "embeddings":
            vector = self._embed(text)
            pts = self.client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
            return pts  # models.ScoredPoint
        else:
            return self._kw.search(text, top_k=top_k)