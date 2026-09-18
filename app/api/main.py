"""FastAPI 앱. 실행: uv run uvicorn app.api.main:app --reload"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.cache import EmbeddingCache, JsonFileCache
from app.domain.models import JobPosting, MatchResult
from app.matching.config import load_config
from app.matching.embedder import CachedEmbedder, OpenAIEmbedder
from app.matching.judge import EmbeddingJudge, RequirementJudge, YearsJudge
from app.matching.lexicon import Lexicon
from app.matching.llm_judge import LlmJudge
from app.matching.service import MatchingService
from app.repository import JsonRepository

load_dotenv()

INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "index.html"

app = FastAPI(title="지원자-공고 매칭 스코어링 엔진")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@lru_cache
def get_repository() -> JsonRepository:
    return JsonRepository()


@lru_cache
def get_service() -> MatchingService:
    config = load_config()
    embedder = CachedEmbedder(OpenAIEmbedder(config.embedding.model), EmbeddingCache())
    text_judge: RequirementJudge = EmbeddingJudge(embedder, config.thresholds, Lexicon.load())
    if config.llm_judge.enabled:
        # 임베딩 판정을 감싸, 충족으로 확정되지 않은 조건만 LLM으로 다시 판정한다.
        text_judge = LlmJudge(text_judge, config.llm_judge, JsonFileCache())
    return MatchingService(text_judge, YearsJudge(config.years), config)


Repository = Annotated[JsonRepository, Depends(get_repository)]
Service = Annotated[MatchingService, Depends(get_service)]


@app.get("/jobs")
def list_jobs(repository: Repository) -> list[JobPosting]:
    return repository.list_jobs()


# OpenAI 호출이 블로킹이므로 async가 아닌 def로 두어 스레드풀에서 실행되게 한다.
@app.get("/jobs/{job_id}/matches")
def get_matches(job_id: str, repository: Repository, service: Service) -> list[MatchResult]:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"공고를 찾을 수 없음: {job_id}")
    try:
        return service.rank(job, repository.list_candidates(job_id))
    except openai.OpenAIError as e:
        # 예외 메시지에 요청 정보가 섞일 수 있어 응답에는 종류만 노출한다.
        raise HTTPException(
            status_code=503,
            detail=f"임베딩 API 호출 실패({type(e).__name__}). .env의 OPENAI_API_KEY 설정을 확인하세요.",
        ) from e
