"""테스트용 결정적 가짜 구현. OpenAI를 호출하지 않는다."""

import hashlib
import re

_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


class FakeEmbedder:
    """단어 겹침 기반 임베딩: 단어마다 고정된 차원에 1을 더한다.

    코사인 유사도는 겹치는 단어가 많을수록 높아진다. 조사까지 한 단어로 보므로
    ('결산과' != '결산') 테스트 문장은 겹칠 단어를 똑같이 적어야 한다.
    """

    model = "fake-word-overlap"

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in set(_TOKEN.findall(text.lower())):
            # 내장 hash()는 실행마다 달라지므로 고정 해시를 쓴다.
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        return vector
