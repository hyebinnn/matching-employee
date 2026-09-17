"""공고/지원자 JSON 파일 로딩.

- 공고: data/jobs/{job_id}.json
- 지원자: data/resumes/{job_id}/{candidate_id}.json (공고별 폴더)

지금은 손으로 만든 픽스처를 읽지만, 이미지 파싱 결과를 같은 스키마로 저장하면 그대로 교체된다.
"""

import json
from pathlib import Path

from app.domain.models import Candidate, JobPosting

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class JsonRepository:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self._jobs_dir = data_dir / "jobs"
        self._resumes_dir = data_dir / "resumes"

    def list_jobs(self) -> list[JobPosting]:
        return [self._load_job(path) for path in sorted(self._jobs_dir.glob("*.json"))]

    def get_job(self, job_id: str) -> JobPosting | None:
        path = self._jobs_dir / f"{job_id}.json"
        # job_id가 경로로 쓰이므로 data/jobs 밖을 가리키는 입력(../ 등)은 거부한다.
        if path.parent != self._jobs_dir or not path.is_file():
            return None
        return self._load_job(path)

    def list_candidates(self, job_id: str) -> list[Candidate]:
        folder = self._resumes_dir / job_id
        if folder.parent != self._resumes_dir or not folder.is_dir():
            return []
        return [Candidate.model_validate(_read_json(path)) for path in sorted(folder.glob("*.json"))]

    def _load_job(self, path: Path) -> JobPosting:
        job = JobPosting.model_validate(_read_json(path))
        # 지원자 폴더를 job.id로 찾으므로, 파일명과 id가 어긋나면 지원자가 조용히 0명이 된다.
        if job.id != path.stem:
            raise ValueError(f"{path.name}: 공고 id({job.id})가 파일명과 다름")
        return job


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
