"""Fetch public reference source and a frozen pretrained metric-depth checkpoint."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/railway_distance_1'
REPORT = ROOT / 'reports/railway_distance_1'


def metadata(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'railway-distance-research'})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def fetch(item):
    name, url = item
    destination = OUT / name
    if not destination.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'railway-distance-research'})
        temporary = destination.with_suffix(destination.suffix + '.part')
        total = 0
        with urllib.request.urlopen(request, timeout=40) as response, temporary.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
                total += len(block)
                if total % (64 * 1024 * 1024) < len(block):
                    print(f'{name}: {total / 1024**2:.0f} MiB', flush=True)
        temporary.replace(destination)
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'file': name, 'url': url, 'bytes': destination.stat().st_size, 'sha256': digest}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    record_path = REPORT / 'asset_revisions.json'
    if record_path.exists():
        revisions = json.loads(record_path.read_text(encoding='utf8'))
    else:
        github = metadata('https://api.github.com/repos/YvanYin/Metric3D/commits/main')
        huggingface = metadata('https://huggingface.co/api/models/JUGGHM/Metric3D')
        revisions = {'source_revision': github['sha'], 'checkpoint_revision': huggingface['sha'],
                     'model': 'metric3d_vit_small', 'checkpoint': 'metric_depth_vit_small_800k.pth'}
        record_path.write_text(json.dumps(revisions, indent=2) + '\n', encoding='utf8')
    commit = revisions['source_revision']
    revision = revisions['checkpoint_revision']
    items = [
        ('metric3d_source.zip', f'https://codeload.github.com/YvanYin/Metric3D/zip/{commit}'),
        ('metric3d_vit_small.pth', f'https://huggingface.co/JUGGHM/Metric3D/resolve/{revision}/{revisions["checkpoint"]}'),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assets = list(pool.map(fetch, items))
    (REPORT / 'assets.json').write_text(json.dumps({'revisions': revisions, 'assets': assets}, indent=2) + '\n', encoding='utf8')
    print(json.dumps(assets, indent=2), flush=True)


if __name__ == '__main__':
    main()
