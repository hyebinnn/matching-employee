"""매칭 엔진의 도메인 모델.

공고 JSON(지금은 손으로 만든 픽스처, 이후 이미지 파싱 결과)은 JobPosting 스키마로 들어온다.
extra="forbid"로 두어 파싱 결과가 스키마와 어긋나면 조용히 무시하지 않고 즉시 실패시킨다.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequirementCategory(StrEnum):
    SKILL = "skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    LANGUAGE = "language"
    LOCATION = "location"
    OTHER = "other"


class RequirementType(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class MatchStatus(StrEnum):
    MET = "met"
    PARTIAL = "partial"
    UNMET = "unmet"


class JudgedBy(StrEnum):
    EMBEDDING = "embedding"
    YEARS = "years"
    LLM = "llm"


class Requirement(_Model):
    id: str
    category: RequirementCategory
    text: str = Field(min_length=1, description="공고 원문 표현")
    type: RequirementType
    min_years: int | None = Field(default=None, ge=0)


class JobPosting(_Model):
    id: str
    title: str
    company: str
    source_url: str | None = None
    requirements: list[Requirement] = Field(min_length=1)

    @model_validator(mode="after")
    def _requirement_ids_unique(self) -> "JobPosting":
        ids = [r.id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError(f"공고 {self.id}: requirement id가 중복됨")
        return self


class Candidate(_Model):
    id: str
    name: str
    resume_text: str = Field(min_length=1, description="자연어 이력서 원문")
    years_of_experience: int | None = Field(default=None, ge=0)


class Judgement(_Model):
    """판정기(RequirementJudge)의 출력. 점수는 모른다 — 점수화는 scorer의 몫."""

    requirement_id: str
    status: MatchStatus
    judged_by: JudgedBy
    reason: str = Field(description="사람이 읽는 판정 사유")
    similarity: float | None = None
    evidence: str | None = Field(default=None, description="근거가 된 이력서 문장")


class RequirementResult(Judgement):
    points: float = Field(ge=0, description="100점 만점 중 이 조건이 기여한 점수")


class MatchResult(_Model):
    candidate_id: str
    candidate_name: str
    score: float = Field(ge=0, le=100)
    applied_cap: float | None = Field(default=None, description="필수 조건 미충족으로 적용된 점수 상한")
    results: list[RequirementResult]
