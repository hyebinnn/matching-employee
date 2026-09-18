"""점수 계산: 가중치 × status 반영 비율 정규화, 필수 충족률에 따른 cap.

config/scoring.yaml의 임시값이 바뀌어도 테스트가 깨지지 않도록 설정은 make_config로 직접 만든다.
"""

import pytest

from app.domain.models import Candidate, JudgedBy, Judgement, MatchStatus, Requirement
from app.matching.config import ScoringConfig
from app.matching.scorer import score_candidate
from tests.fakes import make_config as _config

MET, PARTIAL, UNMET = MatchStatus.MET, MatchStatus.PARTIAL, MatchStatus.UNMET


def _score(required: list[MatchStatus], preferred: list[MatchStatus], config: ScoringConfig | None = None):
    """필수/우대 조건별 status 목록으로 공고와 판정 결과를 만들어 점수를 계산한다."""
    requirements, judgements = [], []
    for type_, statuses in (("required", required), ("preferred", preferred)):
        for i, status in enumerate(statuses):
            req_id = f"{type_}-{i}"
            requirements.append(Requirement(id=req_id, category="other", text=req_id, type=type_))
            judgements.append(Judgement(requirement_id=req_id, status=status, judged_by=JudgedBy.EMBEDDING, reason="-"))
    candidate = Candidate(id="c", name="지원자", resume_text="이력서")
    return score_candidate(candidate, requirements, judgements, config or _config())


def test_all_met_is_100_and_all_unmet_is_0():
    assert _score([MET, MET], [MET]).score == pytest.approx(100)
    assert _score([UNMET, UNMET], [UNMET], _config(cap_enabled=False)).score == 0


def test_weighted_points_for_mixed_statuses():
    # 가중치 합 = 2 + 2 + 1 + 1 = 6
    # 필수 met 2 + 필수 partial 1 + 우대 met 1 + 우대 unmet 0 = 4 → 66.67
    result = _score([MET, PARTIAL], [MET, UNMET], _config(cap_enabled=False))

    assert [r.points for r in result.results] == pytest.approx([100 * 2 / 6, 100 * 1 / 6, 100 * 1 / 6, 0])
    assert result.score == pytest.approx(100 * 4 / 6)
    assert result.score == pytest.approx(sum(r.points for r in result.results))


def test_required_unmet_cannot_outrank_by_filling_preferred():
    # cap이 없으면 A(필수 1개 누락, 우대 전부) 80 > B(필수 전부, 우대 1개) 70 으로 역전된다.
    a = _score([MET, MET, UNMET], [MET, MET, MET, MET])  # 필수 충족률 2/3 = 0.67 → cap 70
    b = _score([MET, MET, MET], [MET, UNMET, UNMET, UNMET])

    assert a.score == 70
    assert a.applied_cap == 70
    assert b.score == pytest.approx(70)
    assert b.score >= a.score


def test_cap_is_not_recorded_when_score_already_below_cap():
    # 필수 충족률 1/2 = 0.5 → cap 70. 원점수 = 2/8 = 25 < 70
    result = _score([MET, UNMET], [UNMET, UNMET, UNMET, UNMET])

    assert result.score == pytest.approx(25)
    assert result.applied_cap is None


def test_lowest_cap_applies_when_ratio_falls_into_several_tiers():
    # 필수 충족률 = (1 + 0.5 + 0) / 3 = 0.5 → 0.8 단계만 해당 → cap 70
    assert _score([MET, PARTIAL, UNMET], [MET] * 6).applied_cap == 70
    # 필수 충족률 = (1 + 0 + 0) / 3 = 0.33 → 0.5와 0.8 단계 모두 해당 → 더 낮은 40
    assert _score([MET, UNMET, UNMET], [MET] * 6).applied_cap == 40


def test_partial_required_lowers_ratio_below_full():
    # 필수 충족률 = (1 + 0.5) / 2 = 0.75 → 0.8 미만이라 cap 70
    # 원점수 = (2 + 1 + 4) / 8 × 100 = 87.5 → 70으로 깎인다
    result = _score([MET, PARTIAL], [MET, MET, MET, MET])

    assert result.score == 70
    assert result.applied_cap == 70


def test_cap_is_not_recorded_when_raw_score_equals_cap():
    # 필수 충족률 2/3 = 0.67 → cap 70. 원점수 = (4 + 3) / 10 × 100 = 70 (부동소수점으로 70.000000000000014)
    # 상한과 같은 값이면 깎인 것이 아니므로 applied_cap을 남기지 않는다.
    result = _score([MET, MET, UNMET], [MET, MET, MET, UNMET])

    assert result.score == pytest.approx(70)
    assert result.applied_cap is None


def test_disabled_cap_and_preferred_unmet_do_not_cap():
    required_unmet = _score([MET, MET, UNMET], [MET, MET, MET, MET], _config(cap_enabled=False))
    preferred_unmet = _score([MET, MET], [UNMET])

    assert required_unmet.applied_cap is None
    assert required_unmet.score == pytest.approx(80)
    assert preferred_unmet.applied_cap is None
    assert preferred_unmet.score == pytest.approx(80)
