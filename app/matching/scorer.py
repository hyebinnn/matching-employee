"""판정 결과 → 점수. 결정적인 순수 함수로만 계산한다 (LLM이 점수를 매기지 않는다).

조건별 points = 유형 가중치 × status 반영 비율 / 전체 가중치 합 × 100
원점수 = points 합. 필수 조건을 충분히 채우지 못하면(충족률 기준) 점수 상한(cap)이 걸린다.
"""

from app.domain.models import (
    Candidate,
    Judgement,
    MatchResult,
    MatchStatus,
    Requirement,
    RequirementResult,
    RequirementType,
)
from app.matching.config import ScoringConfig

# 부동소수점 오차 여유 (원점수와 cap 비교용)
_EPS = 1e-9


def score_candidate(
    candidate: Candidate,
    requirements: list[Requirement],
    judgements: list[Judgement],
    config: ScoringConfig,
) -> MatchResult:
    by_id = {j.requirement_id: j for j in judgements}
    if set(by_id) != {r.id for r in requirements} or len(judgements) != len(requirements):
        raise ValueError(f"지원자 {candidate.id}: 판정 결과가 공고 조건과 일치하지 않음")

    total_weight = sum(_weight(r, config) for r in requirements)
    results = []
    for requirement in requirements:
        judgement = by_id[requirement.id]
        earned = _weight(requirement, config) * _credit(judgement.status, config)
        points = earned / total_weight * 100 if total_weight else 0.0
        results.append(RequirementResult(**judgement.model_dump(), points=points))

    raw_score = min(sum(r.points for r in results), 100.0)  # 부동소수점 합이 100을 살짝 넘는 경우 방지
    cap = _cap(requirements, by_id, config)
    # 원점수와 cap이 같은데 부동소수점 오차로 상한이 걸린 것처럼 기록되지 않게 여유를 둔다.
    capped = cap is not None and raw_score > cap + _EPS

    return MatchResult(
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        score=cap if capped else raw_score,
        applied_cap=cap if capped else None,
        results=results,
    )


def _weight(requirement: Requirement, config: ScoringConfig) -> float:
    if requirement.type == RequirementType.REQUIRED:
        return config.weights.required
    return config.weights.preferred


def _credit(status: MatchStatus, config: ScoringConfig) -> float:
    return getattr(config.status_credit, status.value)


def required_ratio(requirements: list[Requirement], by_id: dict[str, Judgement], config: ScoringConfig) -> float | None:
    """필수 조건 가중치 중 충족한 비율 (met=1, partial=0.5). 필수 조건이 없으면 None."""
    total = sum(config.weights.required for r in requirements if r.type == RequirementType.REQUIRED)
    if not total:
        return None
    earned = sum(
        config.weights.required * _credit(by_id[r.id].status, config)
        for r in requirements
        if r.type == RequirementType.REQUIRED
    )
    return earned / total


def _cap(requirements: list[Requirement], by_id: dict[str, Judgement], config: ScoringConfig) -> float | None:
    """필수 충족률이 걸리는 단계들 중 가장 낮은 상한. 해당 없으면 None."""
    if not config.required_cap.enabled:
        return None
    ratio = required_ratio(requirements, by_id, config)
    if ratio is None:
        return None
    return min((t.cap for t in config.required_cap.tiers if ratio < t.below_ratio), default=None)
