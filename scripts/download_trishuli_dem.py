import urllib.request
import os

os.makedirs("data/trishuli", exist_ok=True)
dem_path = "data/trishuli/copernicus_dem_30m.tif"

if not os.path.exists(dem_path):
    print("Downloading Copernicus DEM GLO-30 for Bhote Koshi - Trishuli corridor...")
    url = (
        "https://portal.opentopography.org/API/globaldem?"
        "demtype=COP30"
        "&south=28.00&north=28.32&west=85.15&east=85.40"
        "&outputFormat=GTiff"
        "&API_Key=b3a926c9fadf4d5794c50d24588f59a6"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'SpaceFloodMap/1.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        with open(dem_path, "wb") as f:
            f.write(data)
    print(f"Downloaded DEM: {len(data)} bytes -> {dem_path}")
else:
    print(f"DEM already exists: {dem_path}")
