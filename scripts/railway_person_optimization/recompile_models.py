"""Recompile accepted new ONNXs in fresh directories and require identical bytes."""
import argparse
from common import *
from compile_model import compile_rect
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
results=[]
for name in ['int8_384x640_railptq_v1','int8_256x1280_railptq_top384_v1']:
    source=OUTPUT/name;info=json.loads((source/'preparation.json').read_text());hw=info['shape_hw'];out=a.output/name;out.mkdir()
    dw,program,plan=compile_rect(source/'allconv_int8.onnx',out,hw)
    graph_same=sha(program/'graph.bin')==sha(source/f'compiled/program_r{hw[1]}/graph.bin')
    plan_same=sha(out/'compiled/resident_plan/resident_plan.bin')==sha(source/'compiled/resident_plan/resident_plan.bin')
    assert graph_same and plan_same,(name,graph_same,plan_same)
    results.append({'candidate':name,'graph_byte_identical':graph_same,'resident_plan_byte_identical':plan_same,'source_sha256':sha(source/'allconv_int8.onnx')})
dump(a.output/'result.json',{'status':'PASS','results':results,'training':False,'recalibration':False})
print(json.dumps(results),flush=True)
