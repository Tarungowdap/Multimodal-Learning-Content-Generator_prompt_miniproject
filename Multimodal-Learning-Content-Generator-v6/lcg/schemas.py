"""Pydantic schemas: the contract for every JSON the pipeline produces."""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Question(BaseModel):
    question: str = Field(min_length=8)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=5)
    source_quote: str = Field(min_length=8, description="Verbatim evidence from the notes")
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("difficulty", mode="before")
    @classmethod
    def _difficulty_tolerant(cls, v):
        # "Medium", " hard " or an unknown label must not invalidate an otherwise good question.
        v = str(v or "").strip().lower()
        return v if v in ("easy", "medium", "hard") else "medium"

    @field_validator("options")
    @classmethod
    def _options_clean(cls, v: list[str]) -> list[str]:
        # Remove answer labels before checking emptiness and uniqueness.
        v = [re.sub(r"^[A-D][\).:]\s*", "", o.strip()).strip() for o in v]
        if any(not o for o in v):
            raise ValueError("options must be non-empty")
        if len({o.lower() for o in v}) != len(v):
            raise ValueError("options must be unique")
        return v

    @property
    def correct_option(self) -> str:
        return self.options[self.answer_index]


class Quiz(BaseModel):
    topic: str
    questions: list[Question] = Field(min_length=1)


class Summary(BaseModel):
    title: str
    summary: str = Field(min_length=30)
    key_points: list[str] = Field(min_length=2, max_length=10)


class Scene(BaseModel):
    caption: str = Field(min_length=3, max_length=90)
    image_subject: str = Field(min_length=3)
    narration: str = Field(min_length=3)
    detail: str = ""   # optional longer explanation shown on the slide


class ScenePlan(BaseModel):
    scenes: list[Scene] = Field(min_length=2, max_length=4)


class QuestionCheck(BaseModel):
    index: int
    quote_found: bool
    answer_supported: Optional[bool] = None   # blind-judge agrees with the key
    evidence_score: float = 0.0               # 0..1 lexical similarity of quote to notes
    passed: bool
    notes: str = ""


class VerificationReport(BaseModel):
    checks: list[QuestionCheck]
    attempts: int = 1
    dropped: int = 0

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.checks) / len(self.checks) if self.checks else 0.0


class LessonPackage(BaseModel):
    topic: str
    llm_provider: str
    summary: Summary
    quiz: Quiz
    verification: VerificationReport
    scenes: list[Scene]
    image_paths: list[str] = []
    image_sources: list[str] = []
    video_path: Optional[str] = None
    video_source: str = ""        # e.g. "text-to-video (gemini)" or "image sequence"
    video_prompt: str = ""        # the standardised prompt sent to the text-to-video model
    sources: list[str] = []
    log: list[str] = []
