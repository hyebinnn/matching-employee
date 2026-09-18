"""조건 충족 여부 판정. 판정기는 점수 정책을 모르고 Judgement만 돌려준다.

나중에 애매 구간(partial 근처)만 LLM으로 재판정하는 구현을 같은 프로토콜로 추가할 수 있다.
"""

import math
from typing import Protocol

from app.domain.models import Candidate, JudgedBy, Judgement, MatchStatus, Requirement
from app.matching.chunker import split_sentences
from app.matching.config import Thresholds, YearsConfig
from app.matching.embedder import Embedder

# 부동소수점 오차 보정 (예: 25 * 0.28 == 7.000000000000001)
_EPS = 1e-9


class RequirementJudge(Protocol):
    def judge(self, requirements: list[Requirement], candidate: Candidate) -> list[Judgement]:
        """requirements와 같은 순서로 판정 결과를 돌려준다."""
        ...


class EmbeddingJudge:
    """조건 문장과 이력서 문장들의 코사인 유사도 최댓값으로 판정한다."""

    def __init__(self, embedder: Embedder, thresholds: Thresholds) -> None:
        self._embedder = embedder
        self._thresholds = thresholds

    def judge(self, requirements: list[Requirement], candidate: Candidate) -> list[Judgement]:
        if not requirements:
            return []
        chunks = split_sentences(candidate.resume_text)
        if not chunks:
            return [
                Judgement(
                    requirement_id=r.id,
                    status=MatchStatus.UNMET,
                    judged_by=JudgedBy.EMBEDDING,
                    reason="이력서에서 비교할 문장을 찾지 못함",
                )
                for r in requirements
            ]

        # 조건과 이력서 문장을 한 번에 임베딩해 API 호출 횟수를 줄인다.
        vectors = self._embedder.embed([r.text for r in requirements] + chunks)
        requirement_vectors, chunk_vectors = vectors[: len(requirements)], vectors[len(requirements) :]

        judgements = []
        for requirement, req_vec in zip(requirements, requirement_vectors, strict=True):
            similarities = [cosine(req_vec, chunk_vec) for chunk_vec in chunk_vectors]
            best = max(range(len(chunks)), key=similarities.__getitem__)
            similarity = similarities[best]
            status, reason = self._classify(similarity)
            judgements.append(
                Judgement(
                    requirement_id=requirement.id,
                    status=status,
                    judged_by=JudgedBy.EMBEDDING,
                    reason=reason,
                    similarity=similarity,
                    evidence=chunks[best],
                )
            )
        return judgements

    def _classify(self, similarity: float) -> tuple[MatchStatus, str]:
        met, partial = self._thresholds.met, self._thresholds.partial
        if similarity >= met:
            return MatchStatus.MET, f"가장 유사한 문장의 유사도 {similarity:.2f} ≥ 충족 기준 {met:.2f}"
        if similarity >= partial:
            return (
                MatchStatus.PARTIAL,
                f"가장 유사한 문장의 유사도 {similarity:.2f}: 부분 충족 기준 {partial:.2f} 이상, 충족 기준 {met:.2f} 미만",
            )
        return (
            MatchStatus.UNMET,
            f"가장 가까운 문장의 유사도 {similarity:.2f} < 부분 충족 기준 {partial:.2f} (충족 근거 아님)",
        )


class YearsJudge:
    """min_years 조건을 임베딩 없이 지원자의 총 경력 연수와 비교한다.

    한계: 직무별 경력(예: '영업 경력 3년')을 구분하지 않고 총 경력으로만 비교한다.
    """

    def __init__(self, config: YearsConfig) -> None:
        self._partial_ratio = config.partial_ratio

    def judge(self, requirements: list[Requirement], candidate: Candidate) -> list[Judgement]:
        return [self._judge_one(r, candidate.years_of_experience) for r in requirements]

    def _judge_one(self, requirement: Requirement, years: int | None) -> Judgement:
        if requirement.min_years is None:
            raise ValueError(f"조건 {requirement.id}: min_years가 없어 연차 판정을 할 수 없음")
        required = requirement.min_years
        partial_floor = required * self._partial_ratio

        if years is None:
            status, reason = MatchStatus.UNMET, "이력서에 총 경력 연수 정보가 없음"
        elif years + _EPS >= required:
            status, reason = MatchStatus.MET, f"총 경력 {years}년 ≥ 요구 {required}년"
        elif years + _EPS >= partial_floor:
            status = MatchStatus.PARTIAL
            reason = f"총 경력 {years}년: 요구 {required}년에는 못 미치나 부분 충족 기준 {partial_floor:g}년 이상"
        else:
            status = MatchStatus.UNMET
            reason = f"총 경력 {years}년 < 부분 충족 기준 {partial_floor:g}년 (요구 {required}년)"

        return Judgement(
            requirement_id=requirement.id,
            status=status,
            judged_by=JudgedBy.YEARS,
            reason=reason,
        )


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm
