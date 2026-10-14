import urllib.request
import urllib.parse
import json

bbox = "28.00,85.15,28.32,85.40"
print(f"Querying places and major highway network: {bbox}")

query = f"""
[out:json][timeout:30];
(
  node["place"~"town|village|hamlet"]({bbox});
  node["amenity"~"hospital|clinic|health_post"]({bbox});
  way["highway"~"trunk|primary|secondary|tertiary"]({bbox});
);
out body;
"""

servers = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]

for server in servers:
    try:
        print(f"Trying server: {server}")
        url = server + "?data=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={'User-Agent': 'SpaceFloodMap/1.0'})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            elements = data.get('elements', [])
            print(f"Success! Elements: {len(elements)}")
            with open("trishuli_osm_sample.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            break
    except Exception as e:
        print(f"Server {server} failed: {e}")
