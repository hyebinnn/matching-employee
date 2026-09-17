"""임베딩 캐시: 같은 입력을 다시 실행하면 API(내부 임베더)를 호출하지 않아야 한다."""

from app.cache import EmbeddingCache
from app.matching.embedder import CachedEmbedder


class CountingEmbedder:
    """호출된 텍스트를 기록하는 가짜 임베더. 벡터는 텍스트 길이로 결정적으로 만든다."""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def test_rerun_with_same_input_does_not_call_embedder(tmp_path):
    texts = ["엑셀 활용 능력", "고객 응대 경험 3년"]
    first_inner = CountingEmbedder()
    first = CachedEmbedder(first_inner, EmbeddingCache(tmp_path)).embed(texts)

    # 프로세스를 다시 띄운 상황: 같은 캐시 디렉토리를 쓰는 새 인스턴스
    rerun_inner = CountingEmbedder()
    second = CachedEmbedder(rerun_inner, EmbeddingCache(tmp_path)).embed(texts)

    assert len(first_inner.calls) == 1
    assert rerun_inner.calls == []
    assert second == first


def test_only_cache_misses_are_embedded_and_order_is_preserved(tmp_path):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, EmbeddingCache(tmp_path))
    embedder.embed(["가"])

    vectors = embedder.embed(["나다", "가", "나다"])

    assert inner.calls[-1] == ["나다"]  # 캐시에 있는 "가"와 중복된 "나다"는 다시 요청하지 않음
    assert vectors == [[2.0, 1.0], [1.0, 1.0], [2.0, 1.0]]


def test_cache_is_not_shared_across_models(tmp_path):
    cache = EmbeddingCache(tmp_path)
    CachedEmbedder(CountingEmbedder("model-a"), cache).embed(["가"])

    other = CountingEmbedder("model-b")
    CachedEmbedder(other, cache).embed(["가"])

    assert other.calls == [["가"]]
