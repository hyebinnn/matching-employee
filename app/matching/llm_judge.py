"""임베딩이 확정하지 못한 조건만 LLM에게 다시 묻는 판정기.

임베딩은 표현이 다른 같은 경험(예: "화장품 단상자 그래픽 설계" ↔ "패키지 디자인 및 감리")을
잘 잡지 못한다. 다시 물을 유사도 구간은 config/scoring.yaml의 llm_judge.band로 정하고,
지금은 하한이 0이라 충족으로 확정되지 않은 조건이 모두 대상이다. 해당 조건은 이력서 전문과
함께 물어 판정을 덮어쓴다. 점수 계산은 여전히 코드가 한다 — LLM은 status와 근거 문장만 돌려준다.

LLM이 지어낸 근거를 막기 위해, 인용한 문장이 이력서 원문에 실제로 있는지 코드로 확인한다
(공고 파싱에서 OCR 원문과 대조하는 것과 같은 방식).
"""

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from app.cache import JsonFileCache
from app.domain.models import Candidate, JudgedBy, Judgement, MatchStatus, Requirement
from app.matching.config import LlmJudgeConfig
from app.matching.judge import RequirementJudge
from app.parsing.verify import verify_text

SYSTEM_PROMPT = """너는 채용 담당자를 대신해 지원자가 채용 조건을 충족하는지 판정한다.

규칙:
1. 판정은 met / partial / unmet 중 하나다.
   - met: 이력서에 조건을 충족한다고 볼 근거가 분명하다. 표현이 달라도 같은 경험이면 met이다.
   - partial: 관련 경험은 있으나 조건에 미치지 못하거나 범위가 좁다.
   - unmet: 근거가 없다.
2. evidence에는 이력서 원문의 문장을 그대로 한 문장 인용한다. 요약하거나 새로 쓰지 않는다.
   근거가 없으면 빈 문자열로 둔다.
3. reason은 왜 그렇게 판정했는지 한 문장으로 한국어로 쓴다.
4. 이력서에 없는 내용을 추측하지 않는다. 직군이나 업계에 대한 선입견으로 판단하지 않는다."""


class _LlmJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    status: MatchStatus
    reason: str
    evidence: str


class _LlmResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[_LlmJudgement]


class LlmJudge:
    """임베딩 판정기를 감싸, band 안에 든 판정만 LLM 결과로 교체한다."""

    def __init__(
        self,
        base: RequirementJudge,
        config: LlmJudgeConfig,
        cache: JsonFileCache | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._base = base
        self._config = config
        self._cache = cache or JsonFileCache()
        self._client = client

    def judge(self, requirements: list[Requirement], candidate: Candidate) -> list[Judgement]:
        judgements = self._base.judge(requirements, candidate)
        by_id = {r.id: r for r in requirements}
        ambiguous = [j for j in judgements if self._is_ambiguous(j)]
        if not ambiguous:
            return judgements

        replacements = self._resolve(
            [by_id[j.requirement_id] for j in ambiguous], candidate, {j.requirement_id: j for j in ambiguous}
        )
        return [replacements.get(j.requirement_id, j) for j in judgements]

    def _is_ambiguous(self, judgement: Judgement) -> bool:
        if judgement.similarity is None:  # 연차 판정 등 임베딩을 쓰지 않은 조건은 건드리지 않는다
            return False
        return self._config.band.low <= judgement.similarity <= self._config.band.high

    def _resolve(
        self, requirements: list[Requirement], candidate: Candidate, fallback: dict[str, Judgement]
    ) -> dict[str, Judgement]:
        cached, to_ask = {}, []
        for requirement in requirements:
            key = self._cache_key(requirement, candidate)
            hit = self._cache.get(key)
            if hit is None:
                to_ask.append(requirement)
            else:
                cached[requirement.id] = _LlmJudgement.model_validate(hit)

        for requirement, judgement in zip(to_ask, self._ask(to_ask, candidate), strict=True):
            self._cache.set(self._cache_key(requirement, candidate), judgement.model_dump(mode="json"))
            cached[requirement.id] = judgement

        return {rid: self._to_judgement(j, candidate, fallback[rid]) for rid, j in cached.items()}

    def _ask(self, requirements: list[Requirement], candidate: Candidate) -> list[_LlmJudgement]:
        if not requirements:
            return []
        if self._client is None:
            self._client = OpenAI()
        conditions = "\n".join(f"- {r.id}: {r.text}" for r in requirements)
        completion = self._client.chat.completions.parse(
            model=self._config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"[채용 조건]\n{conditions}\n\n[지원자 이력서]\n{candidate.resume_text}",
                },
            ],
            response_format=_LlmResponse,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"LLM이 구조화 응답을 주지 않음: {completion.choices[0].message.refusal or '사유 없음'}")

        by_id = {j.requirement_id: j for j in parsed.judgements}
        missing = {r.id for r in requirements} - set(by_id)
        if missing:
            raise RuntimeError(f"LLM 판정 누락: {', '.join(sorted(missing))}")
        return [by_id[r.id] for r in requirements]

    def _to_judgement(self, llm: _LlmJudgement, candidate: Candidate, fallback: Judgement) -> Judgement:
        evidence, note = self._verified_evidence(llm.evidence, candidate.resume_text)
        return Judgement(
            requirement_id=llm.requirement_id,
            status=llm.status,
            judged_by=JudgedBy.LLM,
            reason=f"LLM 판정: {llm.reason}{note} (임베딩 유사도 {fallback.similarity:.2f}: 충족으로 확정되지 않아 재판정)",
            similarity=fallback.similarity,
            evidence=evidence,
        )

    def _verified_evidence(self, evidence: str, resume_text: str) -> tuple[str | None, str]:
        if not evidence.strip():
            return None, ""
        check = verify_text("evidence", evidence, resume_text, self._config.evidence_min_similarity)
        if check.verified:
            return evidence, ""
        return None, " (인용한 근거가 이력서 원문에서 확인되지 않아 제외)"

    def _cache_key(self, requirement: Requirement, candidate: Candidate) -> str:
        return JsonFileCache.key(self._config.model, SYSTEM_PROMPT, requirement.text, candidate.resume_text)
