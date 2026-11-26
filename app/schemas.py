"""Request/response schemas for the mask propagation API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundaryFlow(BaseModel):
    """Bidirectional integer flow for one adjacent-frame boundary.

    Both arrays have shape ``H x W x 2``; the last dimension is
    ``[dy, dx]`` in pixels.
    """

    model_config = ConfigDict(extra="forbid")

    forward: list[list[list[int]]] = Field(
        ..., description="Frame i -> i+1 displacements [dy, dx], HxWx2"
    )
    backward: list[list[list[int]]] = Field(
        ..., description="Frame i+1 -> i displacements [dy, dx], HxWx2"
    )


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_count: int = Field(
        ..., ge=2, le=40, description="Total frames including both anchors (2-40)"
    )
    height: int = Field(..., ge=1, le=128, description="Frame height (<=128)")
    width: int = Field(..., ge=1, le=128, description="Frame width (<=128)")
    first_mask: list[list[int]] = Field(
        ..., description="Binary mask of frame 0, H rows x W cols, entries 0/1"
    )
    last_mask: list[list[int]] = Field(
        ..., description="Binary mask of frame frame_count-1, HxW, entries 0/1"
    )
    flows: list[BoundaryFlow] = Field(
        ...,
        description="One bidirectional flow pair per boundary, length frame_count-1",
    )


class MaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mask: list[list[int]] = Field(
        ..., description="Complete binary mask for this frame, HxW, entries 0/1"
    )


class ErrorDetail(BaseModel):
    detail: str
