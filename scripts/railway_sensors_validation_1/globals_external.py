"""Same whole640 anchors on frozen external sources, without threshold selection."""
from common import *
source=(ROOT/"scripts/railway_sensors_validation_1/supplement_globals.py").read_text()
source=source.replace('for split,manifest in (("joint",REPORT/"joint_validation_v2_rows.jsonl"),("historical",PROTOCOL/"test.jsonl")):',
                      'for split,manifest in (("external",REPORT/"external_new_available.jsonl"),):')
source=source.replace('path=DATA/row["image_id"]','path=OUTPUT/"datasets/OSDaR23_new"/row["image_id"]')
assert 'path=DATA/row' not in source
exec(compile(source,"supplement_globals_external", "exec"))
