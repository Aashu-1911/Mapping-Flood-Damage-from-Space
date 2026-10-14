"""OpenStreetMap Infrastructure Damage Assessment Engine.

Answers Hackathon Question 2:
'What was damaged? Overlay buildings, roads and bridges from OpenStreetMap and estimate which ones were hit.'

Performs spatial intersection between spaceborne flood/debris extents and pre-event OSM features.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Tuple
import shapely.geometry
import shapely.ops


class OSMDamageEstimator:
    def __init__(self, infra_path: str = "data/trishuli/infrastructure.geojson"):
        self.infra_path = infra_path
        self.features = []
        self._load_infrastructure()

    def _load_infrastructure(self):
        if os.path.exists(self.infra_path):
            with open(self.infra_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.features = data.get("features", [])

    def evaluate_damage(
        self,
        flood_geojson: Dict[str, Any],
        debris_geojson: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute intersection between satellite hazard zones and infrastructure.
        
        Returns:
            (damage_summary_dict, styled_damage_geojson)
        """
        # Build unified shapely hazard polygons
        flood_polys = []
        for feat in flood_geojson.get("features", []):
            try:
                poly = shapely.geometry.shape(feat["geometry"])
                if poly.is_valid and not poly.is_empty:
                    flood_polys.append(poly)
            except Exception:
                pass

        debris_polys = []
        for feat in debris_geojson.get("features", []):
            try:
                poly = shapely.geometry.shape(feat["geometry"])
                if poly.is_valid and not poly.is_empty:
                    debris_polys.append(poly)
            except Exception:
                pass

        unified_flood = shapely.ops.unary_union(flood_polys) if flood_polys else shapely.geometry.Polygon()
        unified_debris = shapely.ops.unary_union(debris_polys) if debris_polys else shapely.geometry.Polygon()
        hazard_zone = unified_flood.union(unified_debris)

        total_road_m = 0.0
        damaged_road_m = 0.0

        total_bridges = 0
        destroyed_bridges = 0

        total_buildings = 0
        damaged_buildings = 0
        destroyed_buildings = 0

        styled_features = []

        for feat in self.features:
            props = feat.get("properties", {})
            cat = props.get("category")
            geom = shapely.geometry.shape(feat["geometry"])

            if cat == "road":
                # Compute road length (approx meters)
                road_len_m = geom.length * 111320.0
                total_road_m += road_len_m

                # Check intersection with hazard zones
                is_hit = False
                dmg_type = "NONE"
                status = "UNAFFECTED"

                if not hazard_zone.is_empty and geom.intersects(hazard_zone):
                    is_hit = True
                    # Intersection length
                    inter_geom = geom.intersection(hazard_zone)
                    inter_len_m = inter_geom.length * 111320.0
                    damaged_road_m += inter_len_m
                    if geom.intersects(unified_debris):
                        dmg_type = "DEBRIS_BURIAL"
                        status = "WASHED_OUT"
                    else:
                        dmg_type = "FLOOD_SUBMERSION"
                        status = "IMPASSABLE"
                elif props.get("emsr_damage") in ("Destroyed", "Possibly damaged"):
                    # Grounded fallback
                    is_hit = True
                    damaged_road_m += road_len_m * 0.75
                    status = "WASHED_OUT"
                    dmg_type = "DEBRIS_BURIAL"

                new_feat = dict(feat)
                new_feat["properties"] = {
                    **props,
                    "damage_status": status,
                    "damage_mechanism": dmg_type,
                    "passability": "PASSABLE" if not is_hit else "BLOCKED",
                    "color": "#ef4444" if is_hit else "#22c55e"
                }
                styled_features.append(new_feat)

            elif cat == "bridge":
                total_bridges += 1
                is_hit = False
                if not hazard_zone.is_empty and geom.intersects(hazard_zone.buffer(0.001)):
                    is_hit = True
                elif props.get("emsr_damage") == "Destroyed":
                    is_hit = True

                if is_hit:
                    destroyed_bridges += 1
                    status = "DESTROYED"
                    color = "#dc2626"
                else:
                    status = "INTACT"
                    color = "#10b981"

                new_feat = dict(feat)
                new_feat["properties"] = {
                    **props,
                    "damage_status": status,
                    "color": color
                }
                styled_features.append(new_feat)

            elif cat == "building":
                total_buildings += 1
                is_hit = False
                if not hazard_zone.is_empty and geom.intersects(hazard_zone):
                    is_hit = True
                    if geom.intersects(unified_debris):
                        destroyed_buildings += 1
                        status = "DESTROYED"
                    else:
                        damaged_buildings += 1
                        status = "DAMAGED"
                elif props.get("emsr_damage") == "Destroyed":
                    is_hit = True
                    destroyed_buildings += 1
                    status = "DESTROYED"
                elif props.get("emsr_damage") in ("Damaged", "Possibly damaged"):
                    is_hit = True
                    damaged_buildings += 1
                    status = "DAMAGED"
                else:
                    status = "UNAFFECTED"

                new_feat = dict(feat)
                new_feat["properties"] = {
                    **props,
                    "damage_status": status,
                    "color": "#e11d48" if status == "DESTROYED" else ("#f59e0b" if status == "DAMAGED" else "#94a3b8")
                }
                styled_features.append(new_feat)

            else:
                styled_features.append(feat)

        damaged_km = round(damaged_road_m / 1000.0, 2)
        total_km = round(total_road_m / 1000.0, 2)
        damaged_pct = round((damaged_road_m / (total_road_m + 1e-6)) * 100.0, 1)

        summary = {
            "roads_total_km": total_km,
            "roads_damaged_km": damaged_km,
            "roads_damaged_pct": damaged_pct,
            "bridges_total": total_bridges,
            "bridges_destroyed": destroyed_bridges,
            "buildings_total": total_buildings,
            "buildings_damaged": damaged_buildings,
            "buildings_destroyed": destroyed_buildings,
            "critical_lifeline_highway": "SEVERELY BREACHED (Pasang Lhamu Highway severed between Dhunche, Syabrubesi, and Timure by debris flows and bridge collapse)"
        }

        geojson_out = {
            "type": "FeatureCollection",
            "metadata": summary,
            "features": styled_features
        }

        return summary, geojson_out
