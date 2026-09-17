"""비전 LLM이 뽑은 조건 문장이 OCR 원문에 실제로 있는지 코드로 확인한다.

LLM이 이미지에 없는 조건을 지어내면 OCR 원문 어디에도 비슷한 구간이 없다.
띄어쓰기·기호 차이는 정규화로 없애고, OCR 오인식 몇 글자는 유사도로 허용한다.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_KEEP = re.compile(r"[0-9a-z가-힣]")


@dataclass(frozen=True)
class Verification:
    requirement_id: str
    similarity: float
    verified: bool
    matched_text: str | None  # OCR 원문에서 가장 비슷했던 구간 (정규화 전 원문)


def verify_text(requirement_id: str, text: str, ocr_text: str, min_similarity: float) -> Verification:
    needle, _ = _normalize(text)
    haystack, positions = _normalize(ocr_text)
    if not needle or not haystack:
        return Verification(requirement_id, 0.0, False, None)

    start = haystack.find(needle)
    if start >= 0:
        similarity, end = 1.0, start + len(needle)
    else:
        similarity, start, end = _best_window(needle, haystack)

    matched = ocr_text[positions[start] : positions[end - 1] + 1] if similarity > 0 else None
    return Verification(requirement_id, similarity, similarity >= min_similarity, matched)


def _normalize(text: str) -> tuple[str, list[int]]:
    """비교에 쓸 문자만 남긴 문자열과, 각 문자의 원문 위치."""
    chars, positions = [], []
    for i, ch in enumerate(text.lower()):
        if _KEEP.match(ch):
            chars.append(ch)
            positions.append(i)
    return "".join(chars), positions


def _best_window(needle: str, haystack: str) -> tuple[float, int, int]:
    """needle 길이의 창을 haystack 위로 밀며 가장 비슷한 구간을 찾는다."""
    size = min(len(needle), len(haystack))
    best = (0.0, 0, size)
    for start in range(len(haystack) - size + 1):
        matcher = SequenceMatcher(None, needle, haystack[start : start + size], autojunk=False)
        if matcher.real_quick_ratio() <= best[0] or matcher.quick_ratio() <= best[0]:
            continue
        ratio = matcher.ratio()
        if ratio > best[0]:
            best = (ratio, start, start + size)
    return best
