"""매칭 서비스: 판정기 라우팅 → 점수 → 랭킹이 끝까지 이어지는지."""

from app.domain.models import Candidate, JobPosting, JudgedBy, MatchStatus
from app.matching.judge import EmbeddingJudge, YearsJudge
from app.matching.service import MatchingService
from tests.fakes import FakeEmbedder, make_config

JOB = JobPosting.model_validate(
    {
        "id": "job-store-manager",
        "title": "매장 운영 관리자",
        "company": "테스트 리테일",
        "requirements": [
            {"id": "stock", "category": "experience", "text": "매장 재고 관리 경험", "type": "required"},
            {"id": "years", "category": "experience", "text": "매장 운영 경력 3년 이상", "type": "required", "min_years": 3},
            {"id": "excel", "category": "skill", "text": "엑셀 매출 분석", "type": "preferred"},
        ],
    }
)


def _service() -> MatchingService:
    # FakeEmbedder의 단어 겹침 유사도에 맞춘 임계값
    config = make_config(met=0.7, partial=0.4)
    return MatchingService(EmbeddingJudge(FakeEmbedder(), config.thresholds), YearsJudge(config.years), config)


def _candidate(candidate_id: str, resume: str, years: int | None) -> Candidate:
    return Candidate(id=candidate_id, name=candidate_id, resume_text=resume, years_of_experience=years)


def test_better_matching_candidate_ranks_higher():
    good = _candidate("good", "매장 재고 관리 경험 보유. 엑셀 매출 분석 보고서 작성.", years=5)
    poor = _candidate("poor", "물류 센터 지게차 운전. 야간 교대 근무.", years=1)

    ranked = _service().rank(JOB, [poor, good])

    assert [r.candidate_id for r in ranked] == ["good", "poor"]
    assert ranked[0].score > ranked[1].score
    # 결과는 공고 조건 순서를 따르고, 조건마다 근거가 붙는다.
    assert [r.requirement_id for r in ranked[0].results] == ["stock", "years", "excel"]
    assert all(r.reason for r in ranked[0].results)


def test_min_years_requirement_is_judged_by_years_not_embedding():
    # 이력서에 조건 문장이 그대로 있어도, 실제 경력이 1년이면 임베딩 유사도와 무관하게 unmet이어야 한다.
    candidate = _candidate("copycat", "매장 운영 경력 3년 이상", years=1)

    [result] = _service().rank(JOB, [candidate])
    years_result = next(r for r in result.results if r.requirement_id == "years")

    assert years_result.judged_by == JudgedBy.YEARS
    assert years_result.status == MatchStatus.UNMET
    assert years_result.similarity is None


def test_tied_scores_are_ordered_by_candidate_id():
    resume = "매장 재고 관리 경험 보유."

    ranked = _service().rank(JOB, [_candidate("b", resume, 5), _candidate("a", resume, 5)])

    assert ranked[0].score == ranked[1].score
    assert [r.candidate_id for r in ranked] == ["a", "b"]
