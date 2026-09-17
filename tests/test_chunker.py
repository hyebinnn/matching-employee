"""이력서 문장 분리: evidence로 보여줄 문장의 품질을 결정한다."""

from app.matching.chunker import split_sentences


def test_splits_lines_bullets_and_sentences():
    resume = "- 회계팀에서 근무했습니다. 월말 결산을 담당했습니다.\n• 전산회계 1급 보유\n1) ERP 사용 경험"

    assert split_sentences(resume) == [
        "회계팀에서 근무했습니다.",
        "월말 결산을 담당했습니다.",
        "전산회계 1급 보유",
        "ERP 사용 경험",
    ]


def test_does_not_split_decimal_numbers():
    assert split_sentences("매출 3.5억 규모 거래처 관리. 영어 회화 가능!") == [
        "매출 3.5억 규모 거래처 관리.",
        "영어 회화 가능!",
    ]


def test_drops_separator_lines_and_duplicates():
    resume = "엑셀 능숙\n\n---\n   \n엑셀 능숙"

    assert split_sentences(resume) == ["엑셀 능숙"]
