"""Freeze two metadata quantiles per group; obtain raw references for actual coarse/fine inputs."""
from common import *
import collections
import cv2
from railway_person_optimization.detector import RectDetector

all_rows=[json.loads(s) for s in (REPORT/'joint240_rows.jsonl').read_text().splitlines()]
groups=collections.defaultdict(list)
for i,row in enumerate(all_rows):groups[row['capture_group_id']].append(i)
indices=[v[k] for _,v in sorted(groups.items()) for k in (len(v)//3,2*len(v)//3)]
assert len(indices)==12
out=OUTPUT/'mechanism_reference_v1';out.mkdir(exist_ok=False)
kw={'device':'pc','fast_golden':True,'library':str(ROOT/'output/railway_system_optimization_1/native_pc_v3/native_rs1.dll'),'options':6276}
g=f=None;references=[]
try:
    g=RectDetector(ROOT/'yolo26_optimization_8',ROOT/'output/railway_person_optimization/int8_384x640_railptq_v1',**kw)
    f=RectDetector(ROOT/'yolo26_optimization_8',ROOT/'output/railway_person_optimization/int8_256x1280_railptq_top384_v1',**kw)
    for i in indices:
        row=all_rows[i];image=cv2.imread(str(DATA/row['image_id']))
        saved=json.loads((OUTPUT/'pc_reference_v1'/(key(row['image_id'])+'.json')).read_text())
        fine=[a['crop_xyxy'] for a in saved['variants']['old_adaptive']['actions'] if a['kind']=='fine']
        assert len(fine)==1
        for shape,engine,crop in [('384x640',g,None),('256x1280',f,fine[0])]:
            if crop:
                x,y,x1,y1=crop;source=image[y:y1,x:x1].copy()
            else:source=image
            value=engine.infer(source,.25,return_raw=True)
            file=f'{i:03}_{shape}.json'
            data={'source_index':i,'image_id':row['image_id'],'group':row['capture_group_id'],
                'shape':shape,'crop_xyxy':crop,'decoded_input_sha256':hashlib.sha256(source.tobytes()).hexdigest(),
                'raw':value['raw'].tolist()}
            dump(out/file,data);references.append({'file':file,'sha256':sha(out/file),**{k:data[k] for k in ('source_index','image_id','group','shape','crop_xyxy','decoded_input_sha256')}})
finally:
    if f:f.close()
    if g:g.close()
contract={'sources':12,'groups':6,'profiles':['384x640','256x1280'],'options':[4,4100,6276],
    'resident':[False,True],'rounds':3,'warmups':3,'formal_calls':432,
    'selection':'two metadata quantiles in each frozen joint group; no labels or predictions used for source choice',
    'fine_pixels':'crop chosen by the already frozen old_adaptive policy, not ground truth',
    'scope':'individual engine-call latency and native counters, not a two-call application accuracy-latency point',
    'cumulative_switches':'same v3 library; flags4 is not the historical whole runtime',
    'diagnostic':'separate 20-call options6276|32|64 profile per shape/resident setting; only actual sampled frame exported',
    'references':references}
dump(out/'manifest.json',contract)
dump(REPORT/'mechanism_contract_v1.json',contract)
print(json.dumps({'references':len(references),'formal_calls':432,'manifest_sha256':sha(out/'manifest.json')}))
