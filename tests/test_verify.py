"""OCR 교차 검증: LLM이 지어낸 조건은 걸러내고, OCR 오인식·띄어쓰기 차이는 허용한다."""

from app.parsing.verify import verify_text

OCR_TEXT = """[자격요건]
• 부가세·원천세 신고
  실무 가능자
• 회계 실무 경력 2년 이상
[우대사항]
- 전산회게 1급 소지자
- 엑셀(VLOOKUP) 활용 우수자"""


def _verify(text: str, min_similarity: float = 0.8):
    return verify_text("r", text, OCR_TEXT, min_similarity)


def test_ignores_spacing_punctuation_and_line_breaks():
    # OCR에서는 기호로 붙고 줄바꿈으로 나뉜 문장
    result = _verify("부가세, 원천세 신고 실무 가능자")

    assert result.similarity == 1.0
    assert result.verified


def test_tolerates_small_ocr_misread():
    # OCR이 '계'를 '게'로 잘못 읽음
    result = _verify("전산회계 1급 소지자")

    assert result.verified
    assert result.similarity < 1.0


def test_hallucinated_requirement_is_not_verified():
    result = _verify("정보처리기사 자격증 보유자")

    assert not result.verified


def test_llm_paraphrase_is_not_verified():
    # 원문 표현을 그대로 옮기라고 지시했으므로, 요약·의역은 검증 실패로 사람이 확인하게 한다.
    result = _verify("엑셀 고급 함수 사용 가능")

    assert not result.verified


def test_matched_text_points_to_original_ocr_segment():
    result = _verify("회계 실무 경력 2년 이상")

    assert result.matched_text == "회계 실무 경력 2년 이상"


def test_threshold_comes_from_config():
    assert _verify("전산회계 1급 소지자", min_similarity=0.8).verified
    assert not _verify("전산회계 1급 소지자", min_similarity=0.99).verified
