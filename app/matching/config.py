"""점수 정책 설정. 값은 config/scoring.yaml에 두고, 여기서는 구조와 검증만 담당한다."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scoring.yaml"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddingConfig(_Config):
    model: str = Field(min_length=1)


class Thresholds(_Config):
    met: float = Field(ge=-1, le=1)
    partial: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Thresholds":
        if self.partial > self.met:
            raise ValueError("thresholds.partial은 thresholds.met 이하여야 함")
        return self


class Weights(_Config):
    required: float = Field(gt=0)
    preferred: float = Field(ge=0)


class StatusCredit(_Config):
    met: float = Field(ge=0, le=1)
    partial: float = Field(ge=0, le=1)
    unmet: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "StatusCredit":
        if not self.met >= self.partial >= self.unmet:
            raise ValueError("status_credit은 met >= partial >= unmet 이어야 함")
        return self


class YearsConfig(_Config):
    partial_ratio: float = Field(gt=0, le=1)


class CapRule(_Config):
    enabled: bool
    cap: float = Field(ge=0, le=100)


class RequiredCap(_Config):
    unmet: CapRule
    partial: CapRule


class ScoringConfig(_Config):
    embedding: EmbeddingConfig
    thresholds: Thresholds
    weights: Weights
    status_credit: StatusCredit
    years: YearsConfig
    required_cap: RequiredCap


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ScoringConfig:
    with path.open(encoding="utf-8") as f:
        return ScoringConfig.model_validate(yaml.safe_load(f))
