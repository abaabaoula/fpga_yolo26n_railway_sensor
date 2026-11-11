"""Second raw-label reader and scalar matcher, with no calls to the old parsing/matching functions."""
from common import *
import collections
import csv
import importlib.util
import math
import numpy as np

selected=[json.loads(s) for s in (PRIOR_REPORT/'external_new_available.jsonl').read_text().splitlines()]
raw_root=PRIOR_OUTPUT/'datasets/OSDaR23_new'
label_cache={}
def raw_objects(row):
    sequence,member=row['image_id'].split('/',1)
    if sequence not in label_cache:
        p=next((raw_root/sequence).glob('*_labels.json'))
        assert sha(p)==row['annotation_sha256']
        label_cache[sequence]=json.loads(p.read_text())['openlabel']
    labels=label_cache[sequence]
    candidates=[]
    for fid,frame in labels['frames'].items():
        stream=frame.get('frame_properties',{}).get('streams',{}).get('rgb_center',{})
        if stream.get('uri','').removeprefix('/')==member:candidates.append((fid,frame))
    assert len(candidates)==1
    fid,frame=candidates[0];assert int(fid)==row['frame_index']
    boxes=[];crowds=[]
    for oid,visible in frame.get('objects',{}).items():
        category=labels['objects'][oid]['type']
        for entry in visible.get('object_data',{}).get('bbox',[]):
            if category!='person' or entry.get('coordinate_system')!='rgb_center':continue
            cx,cy,width,height=entry['val']
            boxes.append({'id':oid,'box':[float(cx)-float(width)*.5,float(cy)-float(height)*.5,
                float(cx)+float(width)*.5,float(cy)+float(height)*.5]})
        if category=='crowd':
            crowds.extend(p['val'] for p in visible.get('object_data',{}).get('poly2d',[]) if p.get('coordinate_system')=='rgb_center')
    return boxes,crowds

def overlap(a,b):
    width=max(0.,min(a[2],b[2])-max(a[0],b[0]));height=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=width*height
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/union if union else 0.

def in_polygon(point,values):
    vertices=list(zip(values[::2],values[1::2]));inside=False;x,y=point
    for i,(ax,ay) in enumerate(vertices):
        bx,by=vertices[(i+1)%len(vertices)]
        cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross)<=1e-9 and min(ax,bx)<=x<=max(ax,bx) and min(ay,by)<=y<=max(ay,by):return True
        if (ay>y)!=(by>y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
    return inside

def matches(predictions,gt,threshold):
    used=set();pairs=[]
    for pi in sorted((i for i,p in enumerate(predictions) if p[4]>=threshold),key=lambda i:(-predictions[i][4],i)):
        possible=[(overlap(predictions[pi],b),gi) for gi,b in enumerate(gt) if gi not in used]
        if possible:
            best,gi=max(possible,key=lambda x:(x[0],-x[1]))
            if best>=.5:used.add(gi);pairs.append((pi,gi))
    return pairs

# Scalar implementation is also checked against the previous evaluator; it does not delegate to it.
sys.path.insert(0,str(ROOT/'railway_person_research'))
spec=importlib.util.spec_from_file_location('previous_metrics',ROOT/'scripts/railway_person_research/evaluate_matrix.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
methods=['global','fixed_top','old_adaptive','fixed_focus','roi_score','grid_sahi','ncnn_adaptive','ncnn_whole1280']
records=[];totals={name:collections.Counter() for name in methods};source_checks=[]
for row in selected:
    raw,crowd=raw_objects(row)
    expected=[(o['object_id'],o['bbox_xyxy_raw']) for o in row['objects'] if o['class_name_raw']=='person']
    assert [(o['id'],o['box']) for o in raw]==expected,('RAW_LABEL_MISMATCH',row['image_id'])
    assert crowd==row.get('crowd_ignore_polygons',[])
    source_checks.append({'image_id':row['image_id'],'person_boxes':len(raw),'raw_label_sha256':row['annotation_sha256']})
    cached=json.loads((PRIOR_OUTPUT/'accuracy_external_v1'/(key(row['image_id'])+'.json')).read_text())
    assert cached['source_sha256']==row['image_sha256']
    gt=[o['box'] for o in raw]
    for name in methods:
        preds=cached['variants'][name]['predictions'];pairs=matches(preds,gt,.25)
        found={gi for pi,gi in pairs};matched_predictions={pi for pi,gi in pairs}
        ignored=sum(i not in matched_predictions and p[4]>=.25 and any(in_polygon(((p[0]+p[2])/2,(p[1]+p[3])/2),poly) for poly in crowd) for i,p in enumerate(preds))
        count=sum(p[4]>=.25 for p in preds)
        previous=old.counts(np.asarray(preds,np.float32).reshape(-1,6),row,.25)
        current={'gt':len(gt),'matched':len(pairs),'predictions':count-ignored,
            'unmatched':count-len(pairs)-ignored,'ignored_crowd_predictions':ignored,
            'small_gt':sum(b[3]-b[1]<64 for b in gt),
            'small_matched':sum(gt[i][3]-gt[i][1]<64 for i in found)}
        assert all(current[k]==previous[k] for k in current),('EVALUATOR_DIFFERENCE',name,row['image_id'],current,previous)
        totals[name].update(current)
        for gi,b in enumerate(gt):
            values=[(overlap(p,b),p[4]) for p in preds]
            best_iou=max((v[0] for v in values),default=0.)
            localized_scores=[score for iou,score in values if iou>=.5]
            category=('matched_primary' if gi in found else 'localized_below_score' if localized_scores and max(localized_scores)<.25
                else 'matching_competition' if localized_scores else 'weak_localization' if best_iou>=.1 else 'no_person_candidate_IoU01')
            height=b[3]-b[1]
            records.append({'method':name,'image_id':row['image_id'],'group':row['capture_group_id'],'object_id':raw[gi]['id'],
                'height_px':height,'height_bin':'lt16' if height<16 else '16to32' if height<32 else '32to64' if height<64 else 'ge64',
                'matched':gi in found,'best_person_IoU':best_iou,'best_localized_score':max(localized_scores,default=None),'category':category})
summary={}
for name,t in totals.items():
    subset=[v for v in records if v['method']==name]
    summary[name]={**dict(t),'recall':t['matched']/t['gt'],'UPI':t['unmatched']/len(selected),
        'failure_categories':dict(collections.Counter(v['category'] for v in subset)),
        'height_bins':{b:{'gt':sum(v['height_bin']==b for v in subset),'matched':sum(v['height_bin']==b and v['matched'] for v in subset)} for b in ('lt16','16to32','32to64','ge64')}}
dump(REPORT/'external_second_evaluator.json',{'status':'PASS','images':len(selected),'raw_label_readers_agree':True,
    'scalar_vs_previous_matching_records':len(selected)*len(methods),'GT_instances':len(records)//len(methods),
    'methods':summary,'sources':source_checks,'method_changes':False,'new_blind_test':False,
    'scope':'Second implementation by same implementer, not independent human audit. Existing person candidates only; cannot diagnose other-class errors from these caches.',
    'input_manifest_sha256':sha(PRIOR_REPORT/'external_new_available.jsonl')})
p=REPORT/'tables/external_failures_second_evaluator.csv';p.parent.mkdir(exist_ok=True)
with p.open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps({'status':'PASS','images':len(selected),'GT_instances':len(records)//len(methods),
    'methods':{n:{'matched':t['matched'],'recall':t['recall'],'categories':t['failure_categories']} for n,t in summary.items()}}))
