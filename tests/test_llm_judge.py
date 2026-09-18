"""애매 구간 LLM 재판정: 무엇을 다시 묻는지, 지어낸 근거를 어떻게 막는지, 캐시가 되는지."""

from types import SimpleNamespace

from app.cache import JsonFileCache
from app.domain.models import Candidate, JudgedBy, Judgement, MatchStatus, Requirement
from app.matching.config import Band, LlmJudgeConfig
from app.matching.llm_judge import LlmJudge, _LlmJudgement, _LlmResponse

RESUME = "화장품 단상자와 용기 그래픽을 설계하고, 인쇄소에 입회해 색 교정을 확인했습니다.\n사내 동호회를 운영했습니다."
CANDIDATE = Candidate(id="c", name="지원자", resume_text=RESUME, years_of_experience=8)


def _requirement(req_id: str, text: str = "패키지 디자인 및 감리 경험") -> Requirement:
    return Requirement(id=req_id, category="experience", text=text, type="required")


class StubBaseJudge:
    """임베딩 판정기 대역: requirement id별로 정해진 유사도와 status를 돌려준다."""

    def __init__(self, sims: dict[str, float | None]) -> None:
        self.sims = sims

    def judge(self, requirements, candidate):
        out = []
        for r in requirements:
            sim = self.sims[r.id]
            status = MatchStatus.UNMET if sim is None or sim < 0.65 else MatchStatus.MET
            out.append(
                Judgement(
                    requirement_id=r.id,
                    status=status,
                    judged_by=JudgedBy.YEARS if sim is None else JudgedBy.EMBEDDING,
                    reason="임베딩 판정",
                    similarity=sim,
                    evidence=None if sim is None else "사내 동호회를 운영했습니다.",
                )
            )
        return out


class StubClient:
    """chat.completions.parse 대역."""

    def __init__(self, answers: dict[str, tuple[MatchStatus, str]]) -> None:
        self.answers = answers
        self.asked: list[list[str]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, model, messages, response_format):
        ids = [line.split(":")[0].removeprefix("- ").strip() for line in messages[1]["content"].splitlines() if line.startswith("- ")]
        self.asked.append(ids)
        parsed = _LlmResponse(
            judgements=[
                _LlmJudgement(requirement_id=i, status=self.answers[i][0], reason="근거 확인", evidence=self.answers[i][1])
                for i in ids
            ]
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))])


def _judge(sims, answers, tmp_path, band=(0.35, 0.65)):
    config = LlmJudgeConfig(
        enabled=True, model="stub-model", band=Band(low=band[0], high=band[1]), evidence_min_similarity=0.8
    )
    client = StubClient(answers)
    judge = LlmJudge(StubBaseJudge(sims), config, cache=JsonFileCache(tmp_path), client=client)
    requirements = [_requirement(rid) for rid in sims]
    return judge.judge(requirements, CANDIDATE), client, judge


def test_only_ambiguous_band_is_sent_to_llm(tmp_path):
    sims = {"low": 0.2, "mid": 0.5, "high": 0.9, "years": None}
    results, client, _ = _judge(sims, {"mid": (MatchStatus.MET, "화장품 단상자와 용기 그래픽을 설계하고")}, tmp_path)

    assert client.asked == [["mid"]]  # 확실한 구간과 연차 판정(유사도 없음)은 그대로 둔다
    by_id = {r.requirement_id: r for r in results}
    assert by_id["mid"].judged_by == JudgedBy.LLM
    assert by_id["mid"].status == MatchStatus.MET
    assert [by_id[k].judged_by for k in ("low", "high", "years")] == [
        JudgedBy.EMBEDDING,
        JudgedBy.EMBEDDING,
        JudgedBy.YEARS,
    ]


def test_llm_status_and_evidence_replace_embedding_result(tmp_path):
    results, _, _ = _judge(
        {"mid": 0.5}, {"mid": (MatchStatus.MET, "인쇄소에 입회해 색 교정을 확인했습니다.")}, tmp_path
    )

    [result] = results
    assert result.status == MatchStatus.MET
    assert result.evidence == "인쇄소에 입회해 색 교정을 확인했습니다."
    assert "LLM 판정" in result.reason
    assert result.similarity == 0.5  # 임베딩 유사도는 근거로 남긴다


def test_fabricated_evidence_is_dropped(tmp_path):
    # 이력서에 없는 문장을 근거로 들면 근거를 버리고 reason에 남긴다 (판정 자체는 LLM 결과를 따른다)
    results, _, _ = _judge({"mid": 0.5}, {"mid": (MatchStatus.MET, "포토샵 자격증을 보유했습니다.")}, tmp_path)

    [result] = results
    assert result.evidence is None
    assert "확인되지 않아 제외" in result.reason


def test_llm_result_is_cached_across_instances(tmp_path):
    answers = {"mid": (MatchStatus.PARTIAL, "")}
    first, client_a, _ = _judge({"mid": 0.5}, answers, tmp_path)
    second, client_b, _ = _judge({"mid": 0.5}, answers, tmp_path)

    assert client_a.asked == [["mid"]]
    assert client_b.asked == []  # 같은 (모델, 조건, 이력서) 조합은 다시 묻지 않는다
    assert second[0].status == first[0].status == MatchStatus.PARTIAL
