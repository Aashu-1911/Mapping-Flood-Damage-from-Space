"""Unified End-to-End Spaceborne Flood Damage Intelligence Pipeline.

Orchestrates all Track B modules into a single production pipeline:
- Area & Date input -> Multi-modal Satellite Ingestion
- SAR Change Detection (Sentinel-1 same-orbit track 85)
- Optical Indices & AI Neural Segmentation (Kuro Siwo / Sen1Floods11)
- OSM Pre-event Infrastructure Damage Quantification
- Cut-off Settlement Graph Reachability
- DEM Valley Flow Path Tracing (Bonus)
- Copernicus EMS EMSR927 Validation Benchmark
- Bilingual English/Nepali Situation Report & Rescuer Copilot
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional

from .contracts import (
    DamageSummary,
    CutoffAnalysisResult,
    ValleyTraceResult,
    ValidationMetrics,
    SituationReportResponse,
    CopilotQueryRequest,
    CopilotQueryResponse,
)
from .sar_change_detector import SARChangeDetector
from .optical_detector import OpticalDetector
from .segmentation_model import FloodSegmentationPipeline
from .osm_damage import OSMDamageEstimator
from .cutoff_settlements import CutoffSettlementAnalyzer
from .dem_tracer import ValleyFlowPathTracer
from .emsr927_validation import EMSR927Validator
from .copilot import SituationReportCopilot


class SpaceFloodPipeline:
    def __init__(
        self,
        dem_path: str = "data/trishuli/copernicus_dem_30m.tif",
        infra_path: str = "data/trishuli/infrastructure.geojson",
        emsr_path: str = "data/trishuli/emsr927_ground_truth.geojson"
    ):
        self.dem_path = dem_path
        self.infra_path = infra_path
        self.emsr_path = emsr_path

        # Initialize engines
        self.sar_engine = SARChangeDetector(dem_path=dem_path)
        self.optical_engine = OpticalDetector()
        self.ai_segmenter = FloodSegmentationPipeline()
        self.osm_engine = OSMDamageEstimator(infra_path=infra_path)
        self.cutoff_engine = CutoffSettlementAnalyzer(infra_path=infra_path)
        self.tracer_engine = ValleyFlowPathTracer(dem_path=dem_path, infra_path=infra_path)
        self.validator_engine = EMSR927Validator(ground_truth_path=emsr_path)

        # Cached state from last run
        self.cached_sar_result: Optional[Dict[str, Any]] = None
        self.cached_damage_summary: Optional[DamageSummary] = None
        self.cached_damage_geojson: Optional[Dict[str, Any]] = None
        self.cached_cutoff_result: Optional[CutoffAnalysisResult] = None
        self.cached_cutoff_geojson: Optional[Dict[str, Any]] = None
        self.copilot: Optional[SituationReportCopilot] = None

    def run_full_pipeline(
        self,
        bbox: List[float] = [85.15, 28.00, 85.40, 28.32],
        event_date: str = "2026-08-26",
        orbit_track: int = 85
    ) -> Dict[str, Any]:
        """Execute end-to-end spaceborne mapping pipeline."""
        print(f"--- Running Spaceborne Flood Pipeline for bbox={bbox}, date={event_date} ---")

        # 1. Sentinel-1 SAR change detection
        sar_res = self.sar_engine.process_sar_change(
            bbox=bbox,
            orbit_track=orbit_track,
            pre_date="2026-08-16",
            post_date="2026-08-28"
        )
        self.cached_sar_result = sar_res

        # 2. Sentinel-2 Optical validation
        opt_res = self.optical_engine.process_scene()

        # 3. AI Deep Learning Segmentation evaluation
        ai_metrics = self.ai_segmenter.evaluate_benchmark_accuracy()

        # 4. OpenStreetMap Infrastructure Damage Assessment
        damage_sum_dict, styled_infra_geojson = self.osm_engine.evaluate_damage(
            flood_geojson=sar_res["flood_geojson"],
            debris_geojson=sar_res["debris_geojson"]
        )

        dmg_summary = DamageSummary(
            total_flooded_area_ha=sar_res["water_area_ha"],
            total_debris_area_ha=sar_res["debris_area_ha"],
            total_inundated_area_ha=sar_res["total_impact_area_ha"],
            roads_total_km=damage_sum_dict["roads_total_km"],
            roads_damaged_km=damage_sum_dict["roads_damaged_km"],
            roads_damaged_pct=damage_sum_dict["roads_damaged_pct"],
            bridges_total=damage_sum_dict["bridges_total"],
            bridges_destroyed=damage_sum_dict["bridges_destroyed"],
            buildings_total=damage_sum_dict["buildings_total"],
            buildings_damaged=damage_sum_dict["buildings_damaged"],
            buildings_destroyed=damage_sum_dict["buildings_destroyed"],
            critical_lifeline_highway=damage_sum_dict["critical_lifeline_highway"]
        )
        self.cached_damage_summary = dmg_summary
        self.cached_damage_geojson = styled_infra_geojson

        # 5. Who is cut off? Settlement network graph reachability
        cutoff_res, styled_settlements_geojson = self.cutoff_engine.analyze_cutoff_settlements(
            styled_infra_geojson=styled_infra_geojson
        )
        self.cached_cutoff_result = cutoff_res
        self.cached_cutoff_geojson = styled_settlements_geojson

        # 6. Bonus: Upstream valley flow path tracing
        trace_res = self.tracer_engine.trace_valley_path(
            upstream_lat=28.285,
            upstream_lon=85.385,
            bbox=bbox
        )

        # 7. Copernicus EMS EMSR927 Validation Benchmark
        val_metrics = self.validator_engine.evaluate(
            predicted_flood_geojson=sar_res["flood_geojson"],
            predicted_debris_geojson=sar_res["debris_geojson"],
            damage_summary=damage_sum_dict
        )

        # 8. Initialize Grounded Bilingual Situation-Report Copilot
        self.copilot = SituationReportCopilot(
            damage_summary=dmg_summary,
            cutoff_results=cutoff_res
        )

        # 9. Generate 1-page Situation Report
        sitrep = self.copilot.generate_situation_report()

        return {
            "status": "SUCCESS",
            "event": "August 2026 Northern Nepal Glacier Collapse & Flood Disaster",
            "date": event_date,
            "aoi_bbox": bbox,
            "sar_change_detection": sar_res,
            "optical_validation": opt_res,
            "ai_segmentation_benchmark": ai_metrics,
            "damage_summary": dmg_summary.model_dump(),
            "cutoff_analysis": cutoff_res.model_dump(),
            "bonus_valley_trace": trace_res.model_dump(),
            "emsr927_validation": val_metrics.model_dump(),
            "situation_report": sitrep.model_dump(),
            "layers": {
                "flood_extent": sar_res["flood_geojson"],
                "debris_extent": sar_res["debris_geojson"],
                "infrastructure_damage": styled_infra_geojson,
                "cutoff_settlements": styled_settlements_geojson,
            }
        }

    def query_copilot(self, query: str, language: str = "en") -> CopilotQueryResponse:
        """Query the grounded rescuer assistant."""
        if self.copilot is None:
            # Run pipeline if not already executed
            self.run_full_pipeline()
        return self.copilot.answer_rescuer_query(CopilotQueryRequest(query=query, language=language))

    def trace_valley(self, lat: float, lon: float) -> ValleyTraceResult:
        """Trace downstream path from custom user coordinate."""
        return self.tracer_engine.trace_valley_path(upstream_lat=lat, upstream_lon=lon)


# Global singleton pipeline instance
pipeline_instance = SpaceFloodPipeline()
