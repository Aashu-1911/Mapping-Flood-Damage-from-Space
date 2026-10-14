import urllib.request
import re

req = urllib.request.Request('https://mapping.emergency.copernicus.eu/activations/EMSR927', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as res:
    content = res.read().decode('utf-8', errors='ignore')
    scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', content)
    print("Script tags:", scripts)
    inline = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
    print("Inline scripts count:", len(inline))
    for s in inline:
        if 'EMSR' in s or 'api' in s or 'activation' in s:
            print("Inline match snippet:", s[:300])

# Also check API endpoints directly:
for endpoint in [
    'https://mapping.emergency.copernicus.eu/api/v1/activations/EMSR927',
    'https://mapping.emergency.copernicus.eu/api/activations/EMSR927',
    'https://mapping.emergency.copernicus.eu/backend/api/activations/EMSR927',
    'https://mapping.emergency.copernicus.eu/activations/EMSR927/data',
]:
    try:
        r = urllib.request.Request(endpoint, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(r, timeout=5) as resp:
            print(f"API {endpoint} -> {resp.status}, len={len(resp.read())}")
    except Exception as e:
        print(f"API {endpoint} -> {e}")
