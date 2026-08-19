"""API contract.

The service returns a **decision**, not a picture. A mask is an intermediate; what
an inspection line acts on is pass or escalate, and the number that justifies it.
The mask is available on request for a human who wants to look.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Guarantee(BaseModel):
    alpha: float = Field(..., description="Bound on the risk named by `loss`.")
    loss: str = Field(
        "pixel",
        description="What alpha bounds: 'pixel' the fraction of defect PIXELS "
        "missed, 'instance' the fraction of defect INSTANCES missed. The same "
        "threshold means a different promise under each.",
    )
    threshold: float = Field(..., description="Calibrated mask threshold that delivers it.")
    n_calibration: int
    held_out_fnr: float = Field(
        ..., description="Measured miss rate under `loss` on the held-out split; "
        "should sit under alpha."
    )
    false_alarm_rate: float | None = Field(
        None,
        description="Fraction of DEFECT-FREE parts this threshold escalates, measured on "
        "the control split. None if the category had no defect-free images. Read it "
        "before deploying: the alpha guarantee says nothing about this number.",
    )


class PredictResponse(BaseModel):
    category: str
    decision: Literal["pass", "escalate"] = Field(
        ..., description="'escalate' when the flagged area reaches min_area_frac."
    )
    flagged_fraction: float = Field(
        ..., description="Fraction of the frame the calibrated mask flags."
    )
    min_area_frac: float = Field(
        ..., description="Flagged area at or above which the part is escalated."
    )
    input_size: int
    guarantee: Guarantee
    mask_png: str | None = Field(
        None, description="Base64 PNG of the binary mask, when return_mask=true."
    )


class ModelInfo(BaseModel):
    category: str
    source_run: str
    loss: str
    input_size: int
    threshold: float
    alpha: float
    held_out_fnr: float
    held_out_mask_fraction: float
    n_calibration: int
    n_control: int
    false_alarm_rate: float | None
    min_area_frac: float
