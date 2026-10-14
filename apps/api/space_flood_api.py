"""FastAPI router for Spaceborne Flood Damage Mapping (Track B).

Exposes REST APIs for:
1. End-to-end spaceborne mapping results for Trishuli August 2026 case study.
2. Vector GeoJSON layers (flood extents, debris extents, damaged infrastructure, cut-off settlements).
3. 1-Page Bilingual Situation Report (English & Nepali).
4. Grounded Rescuer Copilot Q&A chat.
5. Upstream DEM valley flow path tracing.
6. Copernicus EMS EMSR927 accuracy metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.space_flood.pipeline import pipeline_instance
from services.space_flood.contracts import (
    DamageSummary,
    CutoffSettlement,
    ValleyTraceResult,
    ValidationMetrics,
    SituationReportResponse,
    CopilotQueryRequest,
    CopilotQueryResponse,
)

router = APIRouter(prefix="/api/v1/space-flood", tags=["Spaceborne Flood Mapping (Track B)"])


class PipelineRunRequest(BaseModel):
    bbox: List[float] = Field(default=[85.15, 28.00, 85.40, 28.32], description="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    event_date: str = Field(default="2026-08-26", description="Disaster event date YYYY-MM-DD")
    orbit_track: int = Field(default=85, description="Sentinel-1 relative orbit track")


class ValleyTraceRequest(BaseModel):
    lat: float = Field(default=28.285, description="Upstream glacier breach latitude")
    lon: float = Field(default=85.385, description="Upstream glacier breach longitude")


@router.get("/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "UP",
        "service": "Spaceborne Flood Damage Intelligence (Track B)",
        "track": "Track B: Mapping Flood Damage from Space",
        "case_study": "August 2026 Northern Nepal Glacier Collapse & Trishuli Debris Flood (EMSR927)",
        "sensors": ["Sentinel-1 SAR C-band (Orbit 85)", "Sentinel-2 MSI Level-2A", "Copernicus WorldDEM-30"],
        "version": "1.0.0"
    }


@router.get("/trishuli-case-study")
def get_trishuli_case_study() -> Dict[str, Any]:
    """Execute / retrieve complete analysis for the August 2026 Trishuli flood case study."""
    return pipeline_instance.run_full_pipeline()


@router.get("/layers")
def get_map_layers() -> Dict[str, Any]:
    """Retrieve GeoJSON map layers for tactical GIS dashboard visualization."""
    if pipeline_instance.cached_sar_result is None:
        pipeline_instance.run_full_pipeline()

    return {
        "flood_extent": pipeline_instance.cached_sar_result["flood_geojson"],
        "debris_extent": pipeline_instance.cached_sar_result["debris_geojson"],
        "infrastructure_damage": pipeline_instance.cached_damage_geojson,
        "cutoff_settlements": pipeline_instance.cached_cutoff_geojson,
    }


@router.get("/situation-report", response_model=SituationReportResponse)
def get_situation_report():
    """Generate 1-page printable situation report in English and Nepali."""
    if pipeline_instance.copilot is None:
        pipeline_instance.run_full_pipeline()
    return pipeline_instance.copilot.generate_situation_report()


@router.post("/copilot/chat", response_model=CopilotQueryResponse)
def chat_with_copilot(req: CopilotQueryRequest):
    """Ask rescuers' operational queries in English or Nepali with grounded facts."""
    return pipeline_instance.query_copilot(query=req.query, language=req.language)


@router.post("/trace-valley", response_model=ValleyTraceResult)
def trace_valley_path(req: ValleyTraceRequest):
    """Bonus feature: trace downstream flood wave using Copernicus 30m DEM."""
    return pipeline_instance.trace_valley(lat=req.lat, lon=req.lon)


@router.get("/validation", response_model=ValidationMetrics)
def get_emsr927_validation():
    """Retrieve accuracy benchmark compared against official Copernicus EMS EMSR927 maps."""
    if pipeline_instance.cached_sar_result is None:
        pipeline_instance.run_full_pipeline()
    return pipeline_instance.validator_engine.evaluate(
        predicted_flood_geojson=pipeline_instance.cached_sar_result["flood_geojson"],
        predicted_debris_geojson=pipeline_instance.cached_sar_result["debris_geojson"],
        damage_summary=pipeline_instance.cached_damage_summary.dict()
    )


@router.get("/ai-segmentation")
def get_ai_segmentation_benchmark() -> Dict[str, Any]:
    """Retrieve AI multi-modal segmentation benchmark metrics (Kuro Siwo / Sen1Floods11)."""
    return pipeline_instance.ai_segmenter.evaluate_benchmark_accuracy()
