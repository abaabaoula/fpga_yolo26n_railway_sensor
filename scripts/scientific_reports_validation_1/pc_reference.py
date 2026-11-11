"""Fresh exact custom integer reference for the 240 frozen sources; no PC speed claim."""
from common import *
import os
os.environ['OMP_NUM_THREADS']='2'
import cv2
from railway_person_optimization.detector import RectDetector
from railway_budget_research.application import analyze
from railway_sensors_validation_1.strategies import scheduler
selected=[json.loads(s) for s in (REPORT/'joint240_rows.jsonl').read_text(encoding='utf8').splitlines()]
out=OUTPUT/'pc_reference_v1';out.mkdir(exist_ok=True)
contract={'source_manifest_sha256':sha(REPORT/'joint240_rows.jsonl'),'score':.25,
    'methods':['global','old_adaptive'],'options':6276,'timing':'PC accuracy/reference only, not ARM latency',
    'library_sha256':sha(ROOT/'output/railway_system_optimization_1/native_pc_v3/native_rs1.dll')}
cp=out/'contract.json'
if cp.exists():assert json.loads(cp.read_text())==contract
else:dump(cp,contract)
kw={'device':'pc','fast_golden':True,'library':str(ROOT/'output/railway_system_optimization_1/native_pc_v3/native_rs1.dll'),'options':6276}
protocol=json.loads((ROOT/'reports/railway_budget_research/protocol_v1.json').read_text())
g=f=None
try:
    g=RectDetector(ROOT/'yolo26_optimization_8',ROOT/'output/railway_person_optimization/int8_384x640_railptq_v1',**kw)
    f=RectDetector(ROOT/'yolo26_optimization_8',ROOT/'output/railway_person_optimization/int8_256x1280_railptq_top384_v1',**kw)
    for index,row in enumerate(selected):
        p=out/(key(row['image_id'])+'.json')
        if p.exists():
            assert json.loads(p.read_text())['contract_sha256']==sha(cp)
            continue
        source=DATA/row['image_id'];assert sha(source)==row['image_sha256']
        image=cv2.imread(str(source));variants={}
        for name in contract['methods']:
            result=analyze(g,f if name!='global' else None,image,scheduler(name,protocol),.25)
            variants[name]={'predictions':result['predictions'].tolist(),'actions':result['actions']}
        dump(p,{'image_id':row['image_id'],'source_sha256':row['image_sha256'],'contract_sha256':sha(cp),'variants':variants})
        dump(REPORT/'pc_reference_progress.json',{'status':'RUNNING','completed':index+1,'total':len(selected)})
        if (index+1)%20==0:print('PC_REFERENCE',index+1,len(selected),flush=True)
finally:
    if f:f.close()
    if g:g.close()
dump(REPORT/'pc_reference_progress.json',{'status':'PASS','completed':len(selected),'total':len(selected)})
