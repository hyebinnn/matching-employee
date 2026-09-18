"""이력서 원문 → 비교 단위 청크. 형태소 분석기 없이 줄바꿈/불릿/문장 끝/절 경계로 자른다.

한국어는 "시각디자인을 전공하고 화장품 브랜드사에서 7년간 패키지 디자인을 담당했습니다"처럼
한 문장에 여러 경력을 담는다. 문장 단위로만 자르면 조건 하나와 비교할 때 나머지 내용이 섞여
유사도가 희석되므로(실측: 전공 조건 0.28), 쉼표와 연결어미에서 절 단위로 더 자른다.
"""

import re

_BULLET = re.compile(r"^\s*(?:[-•·*▪◦]|\d+[.)])\s+")
# 마침표/물음표/느낌표 뒤에 공백이 올 때만 자른다 ("3.5년" 같은 소수점은 자르지 않음).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# 쉼표, 또는 연결어미(-고, -며, -면서, -지만, -는데, -해서/-아서/-어서/-여서) 뒤 공백.
# "고"와 "서"는 앞 글자를 제한한다 (그러지 않으면 "신고", "브랜드사에서" 같은 명사에서도 잘린다).
_CLAUSE_BREAK = re.compile(r"(?<=,)\s+|(?<=[하했되었았였]고)\s+|(?<=며)\s+|(?<=면서)\s+|(?<=지만)\s+|(?<=는데)\s+|(?<=[해아어여]서)\s+")
_HAS_WORD = re.compile(r"[0-9A-Za-z가-힣]")
# 너무 짧은 조각은 의미가 없고 임베딩 비용만 늘린다.
_MIN_CLAUSE_LEN = 6


def split_sentences(text: str) -> list[str]:
    """비교에 쓸 청크 목록. 문장과, 그 문장을 절로 나눈 조각을 함께 담는다.

    문장을 남기는 이유는 절만 남기면 문맥이 끊긴 조각("전공하고")만 근거로 남을 수 있기 때문이고,
    절을 함께 두는 이유는 긴 문장 안에 묻힌 조건을 살리기 위해서다.
    """
    chunks: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        for sentence in _SENTENCE_END.split(line):
            sentence = sentence.strip()
            if not _HAS_WORD.search(sentence):
                continue
            chunks.append(sentence)
            clauses = [c.strip(" ,") for c in _CLAUSE_BREAK.split(sentence)]
            chunks.extend(c for c in clauses if len(c) >= _MIN_CLAUSE_LEN and c != sentence)
    return list(dict.fromkeys(chunks))
