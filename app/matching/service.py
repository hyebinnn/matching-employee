"""공고 + 지원자들 → 랭킹된 MatchResult 목록.

조건을 판정기로 라우팅한다: min_years가 있으면 연차 판정기, 없으면 임베딩 판정기.
외부 의존(임베더, 캐시)은 판정기를 통해 주입받으므로 여기서는 OpenAI를 모른다.
"""

from app.domain.models import Candidate, JobPosting, MatchResult
from app.matching.config import ScoringConfig
from app.matching.judge import RequirementJudge
from app.matching.scorer import score_candidate


class MatchingService:
    def __init__(
        self,
        embedding_judge: RequirementJudge,
        years_judge: RequirementJudge,
        config: ScoringConfig,
    ) -> None:
        self._embedding_judge = embedding_judge
        self._years_judge = years_judge
        self._config = config

    def rank(self, job: JobPosting, candidates: list[Candidate]) -> list[MatchResult]:
        results = [self._match(job, candidate) for candidate in candidates]
        # 점수 내림차순, 동점이면 지원자 id 오름차순 (입력 순서와 무관하게 결정적)
        return sorted(results, key=lambda r: (-r.score, r.candidate_id))

    def _match(self, job: JobPosting, candidate: Candidate) -> MatchResult:
        years_requirements = [r for r in job.requirements if r.min_years is not None]
        text_requirements = [r for r in job.requirements if r.min_years is None]
        judgements = self._years_judge.judge(years_requirements, candidate) + self._embedding_judge.judge(
            text_requirements, candidate
        )
        return score_candidate(candidate, job.requirements, judgements, self._config)
