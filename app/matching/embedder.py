"""텍스트 → 벡터. 판정기는 Embedder 프로토콜에만 의존하고, 테스트에서는 가짜 구현을 주입한다."""

from typing import Protocol

from openai import OpenAI

from app.cache import EmbeddingCache


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """입력 순서대로 벡터를 돌려준다."""
        ...


class OpenAIEmbedder:
    def __init__(self, model: str, client: OpenAI | None = None, batch_size: int = 100) -> None:
        self.model = model
        # client를 넘기지 않으면 OPENAI_API_KEY 환경변수를 사용한다.
        self._client = client or OpenAI()
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(not t.strip() for t in texts):
            raise ValueError("빈 텍스트는 임베딩할 수 없음")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


class CachedEmbedder:
    """캐시에 없는 텍스트만 모아 내부 임베더를 한 번 호출한다."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self.model = inner.model
        self._inner = inner
        self._cache = cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        found: dict[str, list[float]] = {}
        misses: list[str] = []
        for text in dict.fromkeys(texts):  # 순서 유지 중복 제거
            vector = self._cache.get(self.model, text)
            if vector is None:
                misses.append(text)
            else:
                found[text] = vector

        if misses:
            for text, vector in zip(misses, self._inner.embed(misses), strict=True):
                self._cache.set(self.model, text, vector)
                found[text] = vector

        return [found[t] for t in texts]
