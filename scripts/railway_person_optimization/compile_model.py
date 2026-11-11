"""Shared isolated static compiler and independent DMA/memory census."""
import json,sys,subprocess
from common import *

def compile_rect(qdq,out,hw):
    sys.path.insert(0,str(BASE/'tools'))
    from compiler import compile_onnx,enable_integer_depthwise
    compiled=out/'compiled';m=compile_onnx(qdq,compiled);dw=enable_integer_depthwise(m)
    exporter=(BASE/'tools/export_program.py').read_text()
    exporter=exporter.replace("D=pathlib.Path(__file__).resolve().parents[1];R=D.parent",'D=pathlib.Path('+repr(str(BASE))+');R=D.parent')
    exporter=exporter.replace("p.add_argument('--size',type=int,required=True);","p.add_argument('--size',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);")
    exporter=exporter.replace('x=preprocess(image,a.size)', 'x=rect_preprocess(image,(a.height,a.width))[0]')
    exporter="import sys\nsys.path.insert(0,"+repr(str(ROOT))+")\nfrom railway_person_optimization.geometry import preprocess as rect_preprocess\n"+exporter
    entry=compiled/'export_rectangular.py';entry.write_text(exporter,encoding='utf8')
    program=compiled/('program_r'+str(hw[1]))
    subprocess.run([sys.executable,'-B',str(entry),'--size',str(hw[1]),'--height',str(hw[0]),'--width',str(hw[1]),'--model-dir',str(m),'--output',str(program)],check=True)
    meta=json.loads((program/'program.json').read_text());meta['shape_hw']=hw;dump(program/'program.json',meta)
    sys.path.insert(0,str(BASE/'compiler'))
    from resident_plan import make,write
    plan=make(compiled,hw[1],'all');write(plan,compiled/'resident_plan')
    return dw,program,plan

def census(candidate):
    import numpy as np
    info=json.loads((candidate/'preparation.json').read_text());g=json.loads((candidate/'compiled'/('program_r'+str(info['shape_hw'][1]))/'program.json').read_text());sh=g['shapes'];packets=[]
    for n in g['nodes']:
        if n['op']!='QuantConv':continue
        sid,ci,co,k,stride,pad,zx,zw,zo,mode=n['attrs'];_,_,oh,ow=sh[n['outputs'][0]][0];tx=0;tiles=0
        for oc in range(0,co,32):
            c=min(32,co-oc)
            for y in range(0,oh,16):
                for x in range(0,ow,16):
                    hh,ww=min(16,oh-y),min(16,ow-x);hi,wi=(hh-1)*stride+k,(ww-1)*stride+k
                    tx+=76+c*12+(256 if mode==2 else 0)
                    for chunk in range(0,ci,32):
                        cip=(min(32,ci-chunk)+7)//8*8
                        if x==0 and y==0:tx+=(c+7)//8*8*cip*k*k
                        tx+=cip*hi*wi
                    tiles+=1
        packets.append({'sid':sid,'TX_bytes':tx,'RX_bytes':co*oh*ow,'tiles':tiles})
    used={i for n in g['nodes'] for i in n['inputs']+n['outputs']}
    size=lambda v:int(np.prod(v[0]))*np.dtype(v[1]).itemsize
    return {'shape_hw':info['shape_hw'],'all_exported_tensor_bytes':sum(map(size,sh.values())),'executable_tensor_bytes':sum(size(v) for k,v in sh.items() if k in used),'ordinary_TX_payload_bytes':sum(r['TX_bytes'] for r in packets),'ordinary_RX_payload_bytes':sum(r['RX_bytes'] for r in packets),'max_TX_bytes':max(r['TX_bytes'] for r in packets),'max_RX_bytes':max(r['RX_bytes'] for r in packets),'SimpleDMA_BTT_PASS':all(max(r['TX_bytes'],r['RX_bytes'])<8388608 for r in packets),'packets':packets}
