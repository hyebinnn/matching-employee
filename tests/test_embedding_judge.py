"""임베딩 판정기: 가장 유사한 문장을 근거로 고르고, 설정 임계값으로 status를 나눈다."""

from app.domain.models import Candidate, JudgedBy, MatchStatus, Requirement
from app.matching.config import Thresholds
from app.matching.judge import EmbeddingJudge
from tests.fakes import FakeEmbedder

RESUME = "고객 응대 업무를 했습니다. 월말 결산 부가세 신고 담당. 사내 봉사 동아리 활동."


def _requirement(req_id: str, text: str) -> Requirement:
    return Requirement(id=req_id, category="other", text=text, type="required")


def _judge(requirements, resume=RESUME, met=0.9, partial=0.4, embedder=None):
    judge = EmbeddingJudge(embedder or FakeEmbedder(), Thresholds(met=met, partial=partial))
    return judge.judge(requirements, Candidate(id="c", name="지원자", resume_text=resume))


def test_evidence_is_most_similar_sentence_not_first():
    [judgement] = _judge([_requirement("r", "월말 결산 부가세 신고")])

    assert judgement.evidence == "월말 결산 부가세 신고 담당."
    assert judgement.judged_by == JudgedBy.EMBEDDING


def test_status_by_thresholds_and_unmet_is_marked_as_not_evidence():
    # FakeEmbedder 기준 유사도: 4단어 전부 겹침 ≈ 0.89, 2/4 겹침 ≈ 0.45, 겹침 없음 = 0
    met, partial, unmet = _judge(
        [
            _requirement("met", "월말 결산 부가세 신고"),
            _requirement("partial", "월말 결산 재무 보고"),
            _requirement("unmet", "일본어 비즈니스 회화"),
        ],
        met=0.85,
    )

    assert (met.status, partial.status, unmet.status) == (MatchStatus.MET, MatchStatus.PARTIAL, MatchStatus.UNMET)
    # unmet도 가장 가까운 문장은 남기되, reason에서 충족 근거가 아님을 밝힌다.
    assert unmet.similarity is not None
    assert unmet.evidence is not None
    assert "충족 근거 아님" in unmet.reason


def test_thresholds_come_from_config():
    requirement = _requirement("r", "월말 결산 재무 보고")

    assert _judge([requirement], partial=0.4)[0].status == MatchStatus.PARTIAL
    assert _judge([requirement], partial=0.5)[0].status == MatchStatus.UNMET


def test_embeds_requirements_and_sentences_in_one_call():
    embedder = FakeEmbedder()

    _judge([_requirement("a", "고객 응대"), _requirement("b", "부가세 신고")], embedder=embedder)

    assert len(embedder.calls) == 1


def test_resume_without_sentences_is_unmet_without_calling_embedder():
    embedder = FakeEmbedder()

    [judgement] = _judge([_requirement("r", "고객 응대")], resume="---\n\n", embedder=embedder)

    assert judgement.status == MatchStatus.UNMET
    assert judgement.evidence is None
    assert embedder.calls == []
