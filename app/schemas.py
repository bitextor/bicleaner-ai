"""Request/Response schemas for bicleaner-service."""

from pydantic import BaseModel, Field


class ScoreItem(BaseModel):
    """Single scoring request item."""

    id: str = Field(..., description="Unique identifier for tracking")
    source: str = Field(..., description="Source language text")
    target: str = Field(..., description="Target language text")


class BatchScoreRequest(BaseModel):
    """Batch scoring request."""

    requests: list[ScoreItem] = Field(
        ...,
        description="List of source-target pairs to score",
        min_length=1,
    )


class ScoreResult(BaseModel):
    """Single scoring result."""

    id: str = Field(..., description="Matches request id")
    success: bool = Field(..., description="Whether scoring succeeded")
    score: float | None = Field(None, description="Quality score 0-1 (None if failed)")
    error: str | None = Field(None, description="Error message if failed")


class BatchScoreResponse(BaseModel):
    """Batch scoring response."""

    results: list[ScoreResult] = Field(..., description="Scoring results")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Loaded model type")
    model_path: str = Field(..., description="Model path/identifier")
