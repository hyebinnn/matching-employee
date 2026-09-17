"""판정 결과 → 점수. 결정적인 순수 함수로만 계산한다 (LLM이 점수를 매기지 않는다).

조건별 points = 유형 가중치 × status 반영 비율 / 전체 가중치 합 × 100
원점수 = points 합. 필수 조건이 unmet/partial이면 설정된 상한(cap) 중 가장 낮은 값을 넘지 못한다.
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
    capped = cap is not None and raw_score > cap

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


def _cap(requirements: list[Requirement], by_id: dict[str, Judgement], config: ScoringConfig) -> float | None:
    """해당되는 필수 조건 cap 중 가장 낮은 값. 해당 없으면 None."""
    rules = {MatchStatus.UNMET: config.required_cap.unmet, MatchStatus.PARTIAL: config.required_cap.partial}
    caps = [
        rule.cap
        for r in requirements
        if r.type == RequirementType.REQUIRED
        and (rule := rules.get(by_id[r.id].status)) is not None
        and rule.enabled
    ]
    return min(caps, default=None)
