"""Pydantic models for API validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CaseEvent(BaseModel):
    """An event in a synthetic case history."""

    date: str
    type: str
    severity: Optional[float] = None
    violence: Optional[bool] = None
    weapon_mention: Optional[bool] = None
    details: Optional[str] = None


class CaseRequest(BaseModel):
    """A prediction request for one synthetic case."""

    case_id: str = Field(default="api_case")
    start_date: Optional[str] = None
    events: List[CaseEvent]
    outcome: Optional[str] = "unknown"
    risk_category: Optional[str] = "low"

    def to_case(self) -> Dict[str, Any]:
        """Convert the request to the internal case dictionary."""
        return {
            "case_id": self.case_id,
            "start_date": self.start_date,
            "events": [event.model_dump(exclude_none=True) for event in self.events],
            "outcome": self.outcome,
            "risk_category": self.risk_category,
        }


class BatchPredictionRequest(BaseModel):
    """A prediction request for multiple synthetic cases."""

    cases: List[CaseRequest]


class PredictionResponse(BaseModel):
    """API response for one prediction."""

    case_id: str
    risk_score: float
    risk_level: str
    features: Dict[str, float]
    top_factors: List[Dict[str, Any]]
    recommended_actions: List[str]
    disclaimer: str


class BatchPredictionResponse(BaseModel):
    """API response for batch predictions."""

    predictions: List[PredictionResponse]
    disclaimer: str

