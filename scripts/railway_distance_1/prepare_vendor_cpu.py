"""Keep public source unchanged; make a logged CPU-portability working copy."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/railway_distance_1'
REPORT = ROOT / 'reports/railway_distance_1'


def main():
    revisions = json.loads((REPORT / 'asset_revisions.json').read_text())
    source = OUT / 'vendor' / ('Metric3D-' + revisions['source_revision'])
    target = OUT / 'vendor_cpu'
    if not target.exists():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    replacements = {
        'mono/utils/comm.py': [('from mmcv.utils import collect_env as collect_base_env',
                               'from mmengine.utils.dl_utils import collect_env as collect_base_env')],
        'mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py': [
            ('bins_num, device="cuda")', 'bins_num, device=next(self.parameters()).device)')]
    }
    patches, records = [], []
    for name, substitutions in replacements.items():
        old = (source / name).read_text(encoding='utf8')
        new = old
        for before, after in substitutions:
            assert new.count(before) == 1, (name, before)
            new = new.replace(before, after)
        (target / name).write_text(new, encoding='utf8', newline='\n')
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile='original/' + name, tofile='cpu/' + name))
        records.append(dict(file=name, original_sha256=hashlib.sha256((source / name).read_bytes()).hexdigest(),
                            cpu_sha256=hashlib.sha256((target / name).read_bytes()).hexdigest()))
    (REPORT / 'cpu_portability.patch').write_text(''.join(patches), encoding='utf8')
    (REPORT / 'cpu_portability.json').write_text(json.dumps(records, indent=2) + '\n')
    print('CPU working copy ready. Only environment-collection import and constant tensor device changed.')


if __name__ == '__main__':
    main()
