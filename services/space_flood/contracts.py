"""Typed Pydantic contracts for Spaceborne Flood Damage Mapping (Track B)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DamageSummary(BaseModel):
    """Damage summary statistics across infrastructure."""
    total_flooded_area_ha: float = Field(..., description="Estimated flood area in hectares")
    total_debris_area_ha: float = Field(..., description="Estimated debris flow area in hectares")
    total_inundated_area_ha: float = Field(..., description="Total affected area in hectares")
    roads_total_km: float = Field(..., description="Total analyzed road network length in km")
    roads_damaged_km: float = Field(..., description="Damaged or washed-out road length in km")
    roads_damaged_pct: float = Field(..., description="Percentage of road network damaged")
    bridges_total: int = Field(..., description="Total bridge crossings in corridor")
    bridges_destroyed: int = Field(..., description="Number of bridges destroyed or washed out")
    buildings_total: int = Field(..., description="Total analyzed buildings")
    buildings_damaged: int = Field(..., description="Number of buildings damaged or flooded")
    buildings_destroyed: int = Field(..., description="Number of buildings completely destroyed")
    critical_lifeline_highway: str = Field(..., description="Status of Pasang Lhamu Highway (H09)")


class CutoffSettlement(BaseModel):
    """Status of a settlement regarding connectivity to emergency hubs."""
    settlement_id: str
    name: str
    name_ne: str
    place_type: str
    population: int
    coordinates: List[float]
    status: str = Field(..., description="CONNECTED | CUT_OFF | SEVERELY_ISOLATED")
    nearest_hospital: Optional[str] = None
    distance_to_hospital_km: Optional[float] = None
    road_access_intact: bool
    bridge_access_intact: bool
    isolation_reason: str
    priority_level: str = Field(..., description="HIGH | MEDIUM | LOW | NORMAL")


class CutoffAnalysisResult(BaseModel):
    """Aggregate results of the who is cut off analysis."""
    total_settlements: int
    cutoff_settlements_count: int
    isolated_population_total: int
    active_hospitals_accessible: int
    severed_choke_points: List[str]
    cutoff_settlements: List[CutoffSettlement]


class ValleyTracePoint(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    distance_from_origin_km: float


class ValleyTraceResult(BaseModel):
    """Bonus feature: trace flood down the valley using elevation data."""
    origin_point: List[float]
    valley_length_km: float
    flow_path: List[List[float]]
    flow_points: List[ValleyTracePoint]
    downstream_settlements_hit: List[str]
    estimated_arrival_minutes: Dict[str, float]


class ValidationMetrics(BaseModel):
    """Comparison against official Copernicus EMS EMSR927 maps."""
    activation_id: str = "EMSR927"
    event_name: str = "August 2026 Northern Nepal Glacier Collapse & Flood"
    iou_flood_extent: float = Field(..., description="Intersection over Union vs EMSR927 extent")
    precision: float = Field(..., description="Precision score vs ground truth")
    recall: float = Field(..., description="Recall score vs ground truth")
    f1_score: float = Field(..., description="F1-score")
    building_grading_accuracy: float = Field(..., description="Accuracy vs EMSR927 building grading")
    bridge_destruction_match: str = Field(..., description="Match on bridge washouts")


class SituationReportResponse(BaseModel):
    """1-page situation report in English and Nepali."""
    report_id: str
    disaster_event: str
    event_date: str
    report_timestamp: str
    country: str = "Nepal"
    district: str = "Rasuwa (Bhote Koshi - Trishuli Corridor)"
    summary_english: str
    summary_nepali: str
    key_damage_figures: DamageSummary
    cutoff_figures: Dict[str, Any]
    priority_rescue_actions_en: List[str]
    priority_rescue_actions_ne: List[str]
    data_attributions: List[str]
    limitations: List[str]


class CopilotQueryRequest(BaseModel):
    query: str
    language: str = "en"  # "en" or "ne"


class CopilotQueryResponse(BaseModel):
    query: str
    language: str
    answer: str
    grounded_metrics: Dict[str, Any]
    sources_cited: List[str]
