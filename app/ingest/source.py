"""공고 원본(메타데이터 + 본문 이미지) 입력.

지금은 수동으로 내려받은 이미지를 폴더에서 읽는다. 사람인 API 승인 후에는
API로 메타데이터와 상세 페이지 이미지를 받아오는 구현을 같은 프로토콜로 추가한다.

폴더 구조:
    data/raw/{job_id}/meta.json      {"title", "company", "source_url"}
    data/raw/{job_id}/images/*       본문 이미지 (파일명 순서 = 공고 내 순서)
"""

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class RawJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    title: str
    company: str
    source_url: str | None = None
    image_paths: list[Path]


class JobImageSource(Protocol):
    def load(self, job_id: str) -> RawJob: ...


class _Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    company: str
    source_url: str | None = None


class LocalImageSource:
    def __init__(self, root: Path = DEFAULT_RAW_DIR) -> None:
        self._root = root

    def load(self, job_id: str) -> RawJob:
        folder = self._root / job_id
        if folder.parent != self._root or not folder.is_dir():
            raise FileNotFoundError(f"원본 폴더가 없음: {folder}")

        with (folder / "meta.json").open(encoding="utf-8") as f:
            meta = _Meta.model_validate(json.load(f))
        image_paths = sorted(p for p in (folder / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise FileNotFoundError(f"공고 이미지가 없음: {folder / 'images'}")

        return RawJob(job_id=job_id, image_paths=image_paths, **meta.model_dump())
