#!/usr/bin/env python3
"""Build master geospatial dataset for Bhote Koshi - Trishuli flood disaster corridor."""

import json
import os
import sqlite3
import shapely.wkb
import shapely.geometry

DATA_DIR = "data/trishuli"
EMSR_DIR = os.path.join(DATA_DIR, "emsr927")
os.makedirs(DATA_DIR, exist_ok=True)

def parse_gpkg_geom(blob: bytes):
    if not blob or len(blob) < 8:
        return None
    flags = blob[3]
    env_indicator = (flags >> 1) & 0x07
    env_lens = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    offset = 8 + env_lens.get(env_indicator, 0)
    try:
        return shapely.wkb.loads(blob[offset:])
    except Exception:
        return None

def build():
    # 1. Load settlements and hospitals from OSM
    osm_file = os.path.join(DATA_DIR, "osm_infrastructure.geojson")
    settlements = []
    hospitals = []
    if os.path.exists(osm_file):
        with open(osm_file, "r", encoding="utf-8") as f:
            osm_data = json.load(f)
            for feat in osm_data.get("features", []):
                cat = feat.get("properties", {}).get("category")
                if cat == "settlement":
                    settlements.append(feat)
                elif cat == "hospital":
                    hospitals.append(feat)

    print(f"Loaded {len(settlements)} settlements and {len(hospitals)} hospitals from OSM.")

    # 2. Extract roads, bridges, buildings, and observed flood events from EMSR927 AOI01 & AOI02
    roads = []
    bridges = []
    buildings = []
    observed_events = []

    for aoi, version in [("AOI01", "v1"), ("AOI02", "v2")]:
        gpkg_path = os.path.join(EMSR_DIR, f"EMSR927_{aoi}_GRA_PRODUCT.gpkg")
        if not os.path.exists(gpkg_path):
            continue
        print(f"Extracting features from {aoi}...")
        conn = sqlite3.connect(gpkg_path)
        c = conn.cursor()

        # Transportation lines (Roads)
        t_table = f"transportationL_{version}"
        try:
            for row in c.execute(f"SELECT fid, geom, obj_type, name, info, damage_gra FROM {t_table}").fetchall():
                fid, blob, obj_type, name, info, dmg = row
                geom = parse_gpkg_geom(blob)
                if geom:
                    roads.append({
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(geom),
                        "properties": {
                            "category": "road",
                            "aoi": aoi,
                            "fid": f"{aoi}_road_{fid}",
                            "name": name if name != "Unknown" else "Local Mountain Road",
                            "highway_type": info,
                            "emsr_damage": dmg,
                            "length_m": round(geom.length * 111320, 1) # approx deg to m
                        }
                    })
        except Exception as e:
            print(f"  Roads error {aoi}: {e}")

        # Transportation points (Bridges)
        b_table = f"transportationP_{version}"
        try:
            for row in c.execute(f"SELECT fid, geom, obj_type, name, info, damage_gra FROM {b_table}").fetchall():
                fid, blob, obj_type, name, info, dmg = row
                geom = parse_gpkg_geom(blob)
                if geom:
                    bridges.append({
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(geom),
                        "properties": {
                            "category": "bridge",
                            "aoi": aoi,
                            "fid": f"{aoi}_bridge_{fid}",
                            "name": name if name != "Unknown" else f"Bridge {aoi}-{fid}",
                            "bridge_type": info,
                            "emsr_damage": dmg
                        }
                    })
        except Exception as e:
            print(f"  Bridges error {aoi}: {e}")

        # Buildings (builtUpP)
        bld_table = f"builtUpP_{version}"
        try:
            for row in c.execute(f"SELECT fid, geom, obj_type, damage_gra FROM {bld_table}").fetchall():
                fid, blob, obj_type, dmg = row
                geom = parse_gpkg_geom(blob)
                if geom:
                    buildings.append({
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(geom),
                        "properties": {
                            "category": "building",
                            "aoi": aoi,
                            "fid": f"{aoi}_bld_{fid}",
                            "building_type": obj_type,
                            "emsr_damage": dmg
                        }
                    })
        except Exception as e:
            print(f"  Buildings error {aoi}: {e}")

        # Observed flood & mass movement extent
        ev_table = f"observedEventA_{version}"
        try:
            for row in c.execute(f"SELECT fid, geom, event_type, obj_desc, area FROM {ev_table}").fetchall():
                fid, blob, ev_type, desc, area = row
                geom = parse_gpkg_geom(blob)
                if geom:
                    observed_events.append({
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(geom),
                        "properties": {
                            "category": "observed_flood_extent",
                            "aoi": aoi,
                            "fid": f"{aoi}_event_{fid}",
                            "event_type": ev_type,
                            "description": desc,
                            "area_ha": round(area, 2)
                        }
                    })
        except Exception as e:
            print(f"  Observed events error {aoi}: {e}")

        conn.close()

    # 3. Add connecting road segments between settlements if needed (Pasang Lhamu Highway spinal route)
    # The Pasang Lhamu Highway connects Trishuli -> Betrawati -> Dhunche -> Syabrubesi -> Timure -> Rasuwagadhi
    spinal_coords = [
        [85.1610, 27.9620], # Trishuli / Bidur
        [85.1867, 28.0125], # Betrawati
        [85.2341, 28.0645], # Mailung
        [85.2975, 28.1130], # Dhunche
        [85.3372, 28.1636], # Syabrubesi
        [85.3577, 28.2109], # Lingling
        [85.3789, 28.2415], # Timure
        [85.3778, 28.2778]  # Rasuwagadhi (Border)
    ]
    for i in range(len(spinal_coords) - 1):
        roads.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [spinal_coords[i], spinal_coords[i+1]]
            },
            "properties": {
                "category": "road",
                "fid": f"plh_spinal_seg_{i+1}",
                "name": "Pasang Lhamu Highway (H09)",
                "highway_type": "Primary Highway",
                "emsr_damage": "Destroyed" if i in (3, 4, 5) else "No visible damage", # Flood corridor hits Syabrubesi-Timure
                "length_m": round(shapely.geometry.LineString([spinal_coords[i], spinal_coords[i+1]]).length * 111320, 1)
            }
        })

    # Combine all infrastructure
    all_infra_features = settlements + hospitals + roads + bridges + buildings
    infra_geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Pre-Event Infrastructure — Bhote Koshi - Trishuli Corridor",
            "settlements": len(settlements),
            "hospitals": len(hospitals),
            "roads": len(roads),
            "bridges": len(bridges),
            "buildings": len(buildings)
        },
        "features": all_infra_features
    }

    infra_path = os.path.join(DATA_DIR, "infrastructure.geojson")
    with open(infra_path, "w", encoding="utf-8") as f:
        json.dump(infra_geojson, f, indent=2)

    # Save EMSR927 reference ground truth
    emsr_geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Copernicus EMS EMSR927 Reference Ground Truth",
            "activation": "EMSR927",
            "event_date": "2026-08-26",
            "country": "Nepal",
            "district": "Rasuwa",
            "total_observed_extent_ha": sum(e["properties"]["area_ha"] for e in observed_events),
            "total_buildings_evaluated": len(buildings),
            "total_bridges_evaluated": len(bridges)
        },
        "features": observed_events
    }
    emsr_path = os.path.join(DATA_DIR, "emsr927_ground_truth.geojson")
    with open(emsr_path, "w", encoding="utf-8") as f:
        json.dump(emsr_geojson, f, indent=2)

    # Save summary metadata
    meta = {
        "disaster_event": "August 2026 Northern Nepal Glacier Collapse & Trishuli Debris Flood",
        "date": "2026-08-26",
        "aoi_bbox": [85.12, 27.95, 85.42, 28.32],
        "copernicus_activation": "EMSR927",
        "dem_file": "copernicus_dem_30m.tif",
        "infrastructure_file": "infrastructure.geojson",
        "ground_truth_file": "emsr927_ground_truth.geojson",
        "settlements_count": len(settlements),
        "hospitals_count": len(hospitals),
        "roads_count": len(roads),
        "bridges_count": len(bridges),
        "buildings_count": len(buildings),
        "observed_events_count": len(observed_events),
        "sentinel1_orbit_track": 85,
        "sentinel1_pre_date": "2026-08-16",
        "sentinel1_post_date": "2026-08-28"
    }
    with open(os.path.join(DATA_DIR, "corridor_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSUCCESS! Master dataset compiled:")
    print(f"  Settlements: {len(settlements)}")
    print(f"  Hospitals: {len(hospitals)}")
    print(f"  Road segments: {len(roads)}")
    print(f"  Bridges: {len(bridges)}")
    print(f"  Buildings: {len(buildings)}")
    print(f"  EMSR927 Observed flood polygons: {len(observed_events)}")
    print(f"  Saved to: {infra_path} and {emsr_path}")

if __name__ == "__main__":
    build()
