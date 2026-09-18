"""용어 사전: 직군 지식을 코드가 아니라 데이터로 주입해 정확도를 올리는 경로."""

from app.domain.models import Candidate, MatchStatus, Requirement
from app.matching.config import Thresholds
from app.matching.judge import EmbeddingJudge
from app.matching.lexicon import MAX_VARIANTS, Lexicon
from tests.fakes import FakeEmbedder

RESUME = "리액트 기반 SSR 프레임워크로 예약 페이지를 만들었습니다.\n사내 동호회를 운영했습니다."
CANDIDATE = Candidate(id="c", name="지원자", resume_text=RESUME, years_of_experience=4)
REQUIREMENT = Requirement(id="r", category="skill", text="Next.js 기반 프론트엔드 개발 경험", type="required")
LEXICON = Lexicon({"Next.js": ["리액트 기반 SSR 프레임워크"]})


def _judge(lexicon: Lexicon | None):
    judge = EmbeddingJudge(FakeEmbedder(), Thresholds(met=0.4, partial=0.2), lexicon)
    return judge.judge([REQUIREMENT], CANDIDATE)[0]


def test_alias_lifts_similarity_and_status():
    # 이력서가 공고 용어("Next.js") 대신 우회 표현만 쓴 경우
    without = _judge(None)
    with_lexicon = _judge(LEXICON)

    assert without.status == MatchStatus.UNMET  # 사전이 없으면 놓친다
    assert with_lexicon.status == MatchStatus.MET
    assert with_lexicon.similarity > without.similarity


def test_applied_alias_is_shown_in_reason():
    assert "용어 사전 적용(Next.js → 리액트 기반 SSR 프레임워크)" in _judge(LEXICON).reason
    assert "용어 사전" not in _judge(None).reason


def test_no_lexicon_means_no_change():
    # 사전이 없어도(빈 사전) 동작과 결과가 같아야 한다 — 직군 무관 동작이 기본이다.
    assert _judge(Lexicon()).model_dump() == _judge(None).model_dump()


def test_variants_are_capped():
    lexicon = Lexicon({"A": [f"별칭{i}" for i in range(20)]})

    variants = lexicon.variants("A 경험")

    assert len(variants) == MAX_VARIANTS
    assert variants[0] == ("A 경험", None)  # 원문은 항상 포함


def test_loads_and_merges_files(tmp_path):
    (tmp_path / "web.json").write_text('{"terms": {"MS-SQL": ["MSSQL"]}}', encoding="utf-8")
    (tmp_path / "design.json").write_text('{"note": "디자인", "terms": {"감리": ["인쇄 감리"]}}', encoding="utf-8")

    lexicon = Lexicon.load(tmp_path)

    assert [v[1] for v in lexicon.variants("MS-SQL 사용 경험")] == [None, "MS-SQL → MSSQL"]
    assert [v[1] for v in lexicon.variants("감리 경험")] == [None, "감리 → 인쇄 감리"]


def test_missing_directory_is_empty_lexicon(tmp_path):
    assert not Lexicon.load(tmp_path / "없는폴더")
