"""임베딩 파일 캐시.

키는 (모델명, 텍스트)의 sha256. 모델명을 키에 넣어 모델을 바꿨을 때 다른 모델의 벡터를 재사용하지 않게 한다.
파일 하나에 벡터 하나를 저장한다 — 동시 실행에도 전체 캐시 파일이 깨지지 않고, 필요한 것만 읽는다.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "embeddings"


class EmbeddingCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = root

    def get(self, model: str, text: str) -> list[float] | None:
        path = self._path(model, text)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)["embedding"]

    def set(self, model: str, text: str, embedding: list[float]) -> None:
        path = self._path(model, text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 임시 파일에 쓴 뒤 교체해서, 중간에 죽어도 반쯤 쓰인 캐시 파일이 남지 않게 한다.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"model": model, "embedding": embedding}, f)
        os.replace(tmp, path)

    def _path(self, model: str, text: str) -> Path:
        digest = hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()
        model_dir = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        return self.root / model_dir / f"{digest}.json"
