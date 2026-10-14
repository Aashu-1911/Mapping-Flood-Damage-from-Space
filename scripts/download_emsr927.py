import urllib.request
import sqlite3
import os

os.makedirs("data/trishuli/emsr927", exist_ok=True)
gpkg_path = "data/trishuli/emsr927/EMSR927_AOI01_GRA_PRODUCT.gpkg"

url = "https://rapidmapping.emergency.copernicus.eu/backend/EMSR927/AOI01/GRA_PRODUCT/EMSR927_AOI01_GRA_PRODUCT_v1.zip?type=gpkg"
print("Saving GeoPackage directly...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = resp.read()
    with open(gpkg_path, "wb") as f:
        f.write(data)

print(f"Saved {len(data)} bytes to {gpkg_path}")

# Query SQLite contents
conn = sqlite3.connect(gpkg_path)
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall()]
print("GeoPackage layers:", tables)
for t in tables:
    count = c.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
    print(f"  Layer '{t}': {count} features")

conn.close()
