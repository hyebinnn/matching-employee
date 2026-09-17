"""연차 조건: 임베딩 없이 years_of_experience와 min_years를 코드로 비교한다."""

import pytest

from app.domain.models import Candidate, JudgedBy, MatchStatus, Requirement
from app.matching.config import YearsConfig
from app.matching.judge import YearsJudge


def _judge(min_years: int, years: int | None, partial_ratio: float = 0.7):
    requirement = Requirement(
        id="r-years", category="experience", text=f"관련 경력 {min_years}년 이상", type="required", min_years=min_years
    )
    candidate = Candidate(id="c", name="지원자", resume_text="이력서", years_of_experience=years)
    [judgement] = YearsJudge(YearsConfig(partial_ratio=partial_ratio)).judge([requirement], candidate)
    return judgement


@pytest.mark.parametrize(
    ("min_years", "years", "expected"),
    [
        (3, 5, MatchStatus.MET),
        (3, 3, MatchStatus.MET),  # 경계: 요구 연차와 같으면 충족
        (3, 2, MatchStatus.UNMET),  # 2 < 3 * 0.7 = 2.1
        (10, 7, MatchStatus.PARTIAL),  # 경계: 정확히 요구 연차의 70%
        (10, 6, MatchStatus.UNMET),
        (0, 0, MatchStatus.MET),  # 신입 가능 공고
    ],
)
def test_years_compared_by_code(min_years, years, expected):
    judgement = _judge(min_years, years)

    assert judgement.status == expected
    assert judgement.judged_by == JudgedBy.YEARS
    assert judgement.similarity is None


def test_partial_boundary_tolerates_float_error():
    # 25 * 0.28 == 7.000000000000001 — 오차 보정이 없으면 경력 7년이 unmet으로 떨어진다.
    assert _judge(min_years=25, years=7, partial_ratio=0.28).status == MatchStatus.PARTIAL


def test_missing_years_is_unmet_with_reason():
    judgement = _judge(min_years=3, years=None)

    assert judgement.status == MatchStatus.UNMET
    assert "경력 연수 정보가 없음" in judgement.reason


def test_partial_ratio_comes_from_config():
    assert _judge(min_years=10, years=5, partial_ratio=0.5).status == MatchStatus.PARTIAL
    assert _judge(min_years=10, years=5, partial_ratio=0.7).status == MatchStatus.UNMET
