"""점수 계산: 가중치 × status 반영 비율 정규화, 필수 조건 미충족 시 cap.

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
    assert _score([UNMET, UNMET], [UNMET], _config(unmet_cap=False)).score == 0


def test_weighted_points_for_mixed_statuses():
    # 가중치 합 = 2 + 2 + 1 + 1 = 6
    # 필수 met 2 + 필수 partial 1 + 우대 met 1 + 우대 unmet 0 = 4 → 66.67
    result = _score([MET, PARTIAL], [MET, UNMET], _config(partial_cap=False))

    assert [r.points for r in result.results] == pytest.approx([100 * 2 / 6, 100 * 1 / 6, 100 * 1 / 6, 0])
    assert result.score == pytest.approx(100 * 4 / 6)
    assert result.score == pytest.approx(sum(r.points for r in result.results))


def test_required_unmet_cannot_outrank_by_filling_preferred():
    # cap이 없으면 A(필수 1개 누락, 우대 전부) 80 > B(필수 전부, 우대 1개) 70 으로 역전된다.
    a = _score([MET, MET, UNMET], [MET, MET, MET, MET])
    b = _score([MET, MET, MET], [MET, UNMET, UNMET, UNMET])

    assert a.score == 40
    assert a.applied_cap == 40
    assert b.score == pytest.approx(70)
    assert b.score > a.score


def test_cap_is_not_recorded_when_score_already_below_cap():
    # 원점수 = 필수 met 2 / 가중치 합 (2+2+1+1+1+1)=8 → 25 < cap 40
    result = _score([MET, UNMET], [UNMET, UNMET, UNMET, UNMET])

    assert result.score == pytest.approx(25)
    assert result.applied_cap is None


def test_lowest_cap_wins_when_required_unmet_and_partial_both_exist():
    result = _score([MET, PARTIAL, UNMET], [MET, MET, MET, MET, MET, MET])

    assert result.applied_cap == 40


def test_partial_cap_applies_when_only_required_partial():
    # 원점수 = (2 + 2 + 1 + 4) / 10 = 90 → partial cap 70
    result = _score([MET, PARTIAL], [MET, MET, MET, MET])

    assert result.score == 70
    assert result.applied_cap == 70


def test_disabled_cap_and_preferred_unmet_do_not_cap():
    required_unmet = _score([MET, MET, UNMET], [MET, MET, MET, MET], _config(unmet_cap=False))
    preferred_unmet = _score([MET, MET], [UNMET])

    assert required_unmet.applied_cap is None
    assert required_unmet.score == pytest.approx(80)
    assert preferred_unmet.applied_cap is None
    assert preferred_unmet.score == pytest.approx(80)
