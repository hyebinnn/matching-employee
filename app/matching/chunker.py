"""이력서 원문 → 문장 단위 청크. 형태소 분석기 없이 줄바꿈/불릿/문장 끝 기준으로 자른다."""

import re

_BULLET = re.compile(r"^\s*(?:[-•·*▪◦]|\d+[.)])\s+")
# 마침표/물음표/느낌표 뒤에 공백이 올 때만 자른다 ("3.5년" 같은 소수점은 자르지 않음).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HAS_WORD = re.compile(r"[0-9A-Za-z가-힣]")


def split_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        for sentence in _SENTENCE_END.split(line):
            sentence = sentence.strip()
            if _HAS_WORD.search(sentence):
                chunks.append(sentence)
    return list(dict.fromkeys(chunks))
