from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_name: str
    graph_path: str
    draft_config: dict


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_tokens: int = Field(default=50, gt=0, le=1000)
    temperature: float = Field(default=0.9, gt=0.0, le=2.0)
    seed: int | None = Field(default=None, description="Random seed")


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    acceptance_rate: float
    num_accepted: int
    num_rejected: int
    total_tokens: int
