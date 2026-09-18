"""직군·고객사 용어 사전. 직군 지식을 코드가 아니라 데이터로 주입하는 자리다.

같은 것을 가리키는 다른 표현("Next.js" ↔ "리액트 기반 SSR 프레임워크")은 임베딩이 놓치기 쉽다.
사전이 있으면 조건 문장을 대체 표현으로 바꾼 변형을 함께 비교하고, 그중 가장 높은 유사도를 쓴다.
조건 문장에 동의어를 덧붙이지 않는 이유는, 문장이 길어질수록 의미가 희석되기 때문이다.

파일은 data/lexicon/*.json (직군별로 나눠 두고 읽을 때 합친다):

    {
      "note": "웹 개발 직군 용어",
      "terms": {"Next.js": ["리액트 기반 SSR 프레임워크"], "MS-SQL": ["마이크로소프트 SQL 서버"]}
    }

사전이 없으면 아무 일도 하지 않는다 — 직군 무관 동작이 기본이고, 사전은 선택적 강화다.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LEXICON_DIR = Path(__file__).resolve().parents[2] / "data" / "lexicon"
# 조건 하나당 만들 변형 수 상한 (임베딩 호출이 무한정 늘지 않게)
MAX_VARIANTS = 6


class LexiconFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str = ""
    terms: dict[str, list[str]] = Field(default_factory=dict)


class Lexicon:
    def __init__(self, terms: dict[str, list[str]] | None = None) -> None:
        self._terms = terms or {}

    @classmethod
    def load(cls, directory: Path = DEFAULT_LEXICON_DIR) -> "Lexicon":
        merged: dict[str, list[str]] = {}
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            data = LexiconFile.model_validate_json(path.read_text(encoding="utf-8"))
            for term, aliases in data.terms.items():
                merged.setdefault(term, [])
                merged[term].extend(a for a in aliases if a not in merged[term])
        return cls(merged)

    def __bool__(self) -> bool:
        return bool(self._terms)

    def variants(self, text: str) -> list[tuple[str, str | None]]:
        """(비교할 문장, 적용된 대체 표현) 목록. 첫 항목은 항상 원문이다."""
        out: list[tuple[str, str | None]] = [(text, None)]
        lowered = text.lower()
        for term, aliases in self._terms.items():
            if term.lower() not in lowered:
                continue
            for alias in aliases:
                if len(out) >= MAX_VARIANTS:
                    return out
                out.append((_replace_ignorecase(text, term, alias), f"{term} → {alias}"))
        return out


def _replace_ignorecase(text: str, term: str, alias: str) -> str:
    index = text.lower().find(term.lower())
    return text[:index] + alias + text[index + len(term) :]
