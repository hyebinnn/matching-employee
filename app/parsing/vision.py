"""비전 LLM으로 공고 이미지에서 조건을 구조화한다.

응답은 pydantic 스키마로 강제하고, id 부여와 검증은 코드가 한다.
같은 (모델, 프롬프트, 이미지) 조합은 파일 캐시로 재사용해 다시 과금되지 않게 한다.
"""

import base64
import hashlib
import io
import json
from pathlib import Path

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict

from app.domain.models import RequirementCategory, RequirementType

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "parse"

SYSTEM_PROMPT = """너는 채용공고 이미지에서 지원자에게 요구하는 조건만 추출한다.

규칙:
1. 지원자가 갖춰야 하거나 갖추면 유리한 조건만 추출한다. 담당업무, 회사 소개, 복리후생, 근무시간, 급여, 전형 절차, 제출 서류는 추출하지 않는다.
2. text에는 이미지에 적힌 표현을 그대로 옮긴다. 요약하거나 다른 말로 바꾸지 않는다. 글머리 기호(-, •, ㆍ, ※ 등)만 뺀다.
3. 원문의 한 항목(한 줄)을 하나의 조건으로 둔다. 한 항목 안에 여러 내용이 나열되어 있어도 나누지 않는다.
4. type: 자격요건·지원자격·필수 영역의 조건은 required, 우대사항·우대·가산점 영역의 조건은 preferred. 영역 구분이 없으면 '우대' 같은 문장 표현으로 판단한다.
5. category: skill(도구·기술·업무 역량), experience(경력·업무 경험), education(학력·전공), certification(자격증·면허), language(어학), location(근무지·거주지·출퇴근 조건), other(그 외).
6. min_years: '경력 N년 이상'처럼 필요한 경력 연수가 숫자로 명시된 경우에만 N을 넣는다. 연수가 없거나 '신입', '경력무관'이면 null이다.
7. '경력무관', '학력무관'처럼 아무 조건도 요구하지 않는 항목은 추출하지 않는다.
8. 이미지 여러 장은 한 공고를 위에서 아래로 자른 조각이고 인접 조각은 조금 겹친다. 겹친 부분 때문에 중복된 조건은 한 번만 추출한다.
9. 이미지에 없는 내용은 절대 만들지 않는다."""


class ExtractedRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: RequirementCategory
    text: str
    type: RequirementType
    min_years: int | None


class ExtractedPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[ExtractedRequirement]


class VisionParser:
    def __init__(self, model: str, client: OpenAI | None = None, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.model = model
        self._client = client
        self._cache_dir = cache_dir

    def parse(self, tiles: list[Image.Image]) -> list[ExtractedRequirement]:
        encoded = [_png_bytes(tile) for tile in tiles]
        cache_path = self._cache_dir / f"{self._cache_key(encoded)}.json"
        if cache_path.exists():
            return ExtractedPosting.model_validate_json(cache_path.read_text(encoding="utf-8")).requirements

        if self._client is None:
            self._client = OpenAI()
        completion = self._client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"채용공고 이미지 조각 {len(tiles)}장 (위에서 아래 순서)"},
                        *(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(b).decode()}", "detail": "high"},
                            }
                            for b in encoded
                        ),
                    ],
                },
            ],
            response_format=ExtractedPosting,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"비전 모델이 구조화 응답을 주지 않음: {message.refusal or '사유 없음'}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(message.parsed.model_dump_json(indent=2), encoding="utf-8")
        return message.parsed.requirements

    def _cache_key(self, encoded_tiles: list[bytes]) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([self.model, SYSTEM_PROMPT]).encode("utf-8"))
        for tile in encoded_tiles:
            digest.update(hashlib.sha256(tile).digest())
        return digest.hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
