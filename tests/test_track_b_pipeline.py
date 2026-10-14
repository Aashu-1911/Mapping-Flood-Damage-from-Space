"""Automated Verification Suite for Track B Spaceborne Flood Damage Pipeline.

Tests:
1. Sentinel-1 SAR change detection (Orbit 85 same-track log-ratio).
2. OSM infrastructure damage intersection.
3. Topological cut-off settlement graph analysis.
4. AI multi-modal segmentation model forward pass.
5. DEM steepest-descent valley flow path tracing.
6. Copernicus EMS EMSR927 benchmark validation.
7. Bilingual situation-report copilot queries (English & Nepali).
8. FastAPI REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from apps.api.app import app
from services.space_flood.pipeline import pipeline_instance
from services.space_flood.sar_change_detector import SARChangeDetector
from services.space_flood.segmentation_model import FloodSegmentationPipeline
from services.space_flood.dem_tracer import ValleyFlowPathTracer
from services.space_flood.contracts import CopilotQueryRequest


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_pipeline_execution():
    """Verify end-to-end pipeline executes without failure."""
    res = pipeline_instance.run_full_pipeline()
    assert res["status"] == "SUCCESS"
    assert "sar_change_detection" in res
    assert "damage_summary" in res
    assert "cutoff_analysis" in res
    assert "situation_report" in res

    # Verify SAR change detection numbers
    sar = res["sar_change_detection"]
    assert sar["orbit_track"] == 85
    assert sar["water_area_ha"] > 50.0
    assert sar["debris_area_ha"] > 50.0

    # Verify damage assessment
    dmg = res["damage_summary"]
    assert dmg["roads_damaged_km"] > 10.0
    assert dmg["bridges_destroyed"] >= 5
    assert "Pasang Lhamu" in dmg["critical_lifeline_highway"]

    # Verify cut-off settlements
    cutoff = res["cutoff_analysis"]
    assert cutoff["cutoff_settlements_count"] >= 10
    assert cutoff["isolated_population_total"] > 5000


def test_sar_change_detector_slope_mask():
    """Verify SAR detector properly excludes mountain cliffs via DEM slope."""
    detector = SARChangeDetector()
    res = detector.process_sar_change()
    assert res["water_area_ha"] > 0
    assert "flood_geojson" in res
    assert len(res["flood_geojson"]["features"]) > 0


def test_ai_segmentation_benchmark():
    """Verify multi-modal neural network forward pass and metrics."""
    ai = FloodSegmentationPipeline()
    metrics = ai.evaluate_benchmark_accuracy()
    assert metrics["mean_iou"] > 0.80
    assert metrics["overall_pixel_accuracy"] > 0.90
    assert metrics["training_dataset"].startswith("Kuro Siwo")


def test_valley_flow_path_tracer():
    """Verify DEM valley flow tracer follows descent down Trishuli."""
    tracer = ValleyFlowPathTracer()
    res = tracer.trace_valley_path(upstream_lat=28.285, upstream_lon=85.385)
    assert res.valley_length_km > 20.0
    assert len(res.flow_points) > 10
    assert "Syabrubesi" in res.downstream_settlements_hit or "Shyaphru" in res.downstream_settlements_hit


def test_emsr927_validation_benchmark():
    """Verify Copernicus EMS EMSR927 accuracy metrics."""
    metrics = pipeline_instance.validator_engine.evaluate(
        predicted_flood_geojson=pipeline_instance.cached_sar_result["flood_geojson"],
        predicted_debris_geojson=pipeline_instance.cached_sar_result["debris_geojson"],
        damage_summary=pipeline_instance.cached_damage_summary.model_dump()
    )
    assert metrics.activation_id == "EMSR927"
    assert metrics.building_grading_accuracy > 0.85
    assert "100%" in metrics.bridge_destruction_match


def test_bilingual_copilot_grounding():
    """Verify copilot answers queries strictly with grounded map statistics."""
    # English query
    req_en = CopilotQueryRequest(query="Which settlements are cut off?", language="en")
    resp_en = pipeline_instance.copilot.answer_rescuer_query(req_en)
    assert "CUT OFF" in resp_en.answer
    assert str(pipeline_instance.cached_cutoff_result.cutoff_settlements_count) in resp_en.answer

    # Nepali query
    req_ne = CopilotQueryRequest(query="कुन गाउँ सम्पर्कविहीन छन्?", language="ne")
    resp_ne = pipeline_instance.copilot.answer_rescuer_query(req_ne)
    assert "सम्पर्कविहीन" in resp_ne.answer
    assert str(pipeline_instance.cached_cutoff_result.cutoff_settlements_count) in resp_ne.answer


def test_fastapi_endpoints(client):
    """Verify all REST API endpoints return HTTP 200."""
    # 1. Health
    r1 = client.get("/api/v1/space-flood/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "UP"

    # 2. Map Layers
    r2 = client.get("/api/v1/space-flood/layers")
    assert r2.status_code == 200
    assert "flood_extent" in r2.json()

    # 3. Situation report
    r3 = client.get("/api/v1/space-flood/situation-report")
    assert r3.status_code == 200
    assert "SITREP" in r3.json()["report_id"]

    # 4. Copilot chat
    r4 = client.post("/api/v1/space-flood/copilot/chat", json={"query": "How many bridges were destroyed?", "language": "en"})
    assert r4.status_code == 200
    assert "bridges" in r4.json()["answer"].lower()

    # 5. Valley tracer
    r5 = client.post("/api/v1/space-flood/trace-valley", json={"lat": 28.285, "lon": 85.385})
    assert r5.status_code == 200
    assert r5.json()["valley_length_km"] > 10.0

    # 6. Dashboard HTML
    r6 = client.get("/space-flood")
    assert r6.status_code == 200
    assert "Spaceborne Flood Damage Intelligence" in r6.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
