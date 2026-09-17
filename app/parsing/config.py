"""공고 이미지 파싱 설정. 값은 config/parsing.yaml에 둔다."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "parsing.yaml"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VisionConfig(_Config):
    model: str | None = None


class TilingConfig(_Config):
    max_height: int = Field(gt=0)
    overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_tile(self) -> "TilingConfig":
        if self.overlap >= self.max_height:
            raise ValueError("tiling.overlap은 tiling.max_height보다 작아야 함")
        return self


class OcrConfig(_Config):
    lang: str = Field(min_length=1)
    upscale: float = Field(ge=1)
    psm: int = Field(ge=0, le=13)


class VerifyConfig(_Config):
    min_similarity: float = Field(gt=0, le=1)


class ParsingConfig(_Config):
    vision: VisionConfig
    tiling: TilingConfig
    ocr: OcrConfig
    verify: VerifyConfig


def load_parsing_config(path: Path = DEFAULT_CONFIG_PATH) -> ParsingConfig:
    with path.open(encoding="utf-8") as f:
        return ParsingConfig.model_validate(yaml.safe_load(f))
