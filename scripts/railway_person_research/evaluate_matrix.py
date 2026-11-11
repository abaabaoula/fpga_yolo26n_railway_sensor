"""Strict annotation-relative evaluation, independent of inference code."""
import argparse,contextlib,hashlib,io,json,sys
from pathlib import Path
import numpy as np


def load_rows(root,plan):
    rows=[json.loads(l) for l in (root/plan['manifest']).read_text(encoding='utf8').splitlines()]
    if plan.get('ids'):
        lookup={r['image_id']:r for r in rows};rows=[lookup[i] for i in json.loads((root/plan['ids']).read_text(encoding='utf8'))]
    return rows


def counts(predictions,row,score):
    from evaluation import match_predictions,height_group
    import cv2
    gt=[o['bbox_xyxy_raw'] for o in row['objects'] if o['class_name_raw']=='person'];m=match_predictions(predictions,gt,score,.5);matched_pred={p for p,g in m['assignments']};ignored=0
    for i,b in enumerate(predictions):
        if i in matched_pred or b[4]<score:continue
        point=((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
        if any(cv2.pointPolygonTest(np.asarray(poly,np.float32).reshape(-1,2),point,False)>=0 for poly in row.get('crowd_ignore_polygons',[])):ignored+=1
    small={i for i,b in enumerate(gt) if b[3]-b[1]<64}
    result=dict(gt=len(gt),matched=m['matched'],predictions=m['predictions']-ignored,unmatched=m['unmatched_predictions']-ignored,ignored_crowd_predictions=ignored,small_gt=len(small),small_matched=len(small&m['matched_gt']))
    for group in ['<32','32-64','64-128','>=128']:
        ids={i for i,b in enumerate(gt) if height_group(b)==group};result['gt_'+group]=len(ids);result['matched_'+group]=len(ids&m['matched_gt'])
    return result


def rates(total,n):
    total=dict(total);total.update(recall=total['matched']/total['gt'] if total['gt'] else None,small_recall=total['small_matched']/total['small_gt'] if total['small_gt'] else None,UPI=total['unmatched']/n,annotation_relative_precision=total['matched']/total['predictions'] if total['predictions'] else None)
    total['height_recall']={g:total['matched_'+g]/total['gt_'+g] if total['gt_'+g] else None for g in ['<32','32-64','64-128','>=128']};return total


def coco_metrics(root,rows,predictions):
    sys.path[:0]=[str(root/'third_party'),str(root/'envs/stage_d_pc_extras')]
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt={'info':{},'licenses':[],'images':[],'annotations':[],'categories':[{'id':1,'name':'person'}]};pred=[];aid=0
    for i,row in enumerate(rows,1):
        gt['images'].append({'id':i,'width':row['width'],'height':row['height']})
        for o in row['objects']:
            if o['class_name_raw']!='person':continue
            x,y,x1,y1=o['bbox_xyxy_raw'];aid+=1;gt['annotations'].append({'id':aid,'image_id':i,'category_id':1,'bbox':[x,y,x1-x,y1-y],'area':(x1-x)*(y1-y),'iscrowd':0})
        for polygon in row.get('crowd_ignore_polygons',[]):
            from pycocotools import mask as maskutil
            rle=maskutil.merge(maskutil.frPyObjects([polygon],row['height'],row['width']));bbox=maskutil.toBbox(rle).tolist();aid+=1;gt['annotations'].append({'id':aid,'image_id':i,'category_id':1,'bbox':bbox,'area':float(maskutil.area(rle)),'iscrowd':1,'segmentation':rle})
        for b in predictions[row['image_id']]:
            x,y,x1,y1,s,_=map(float,b);pred.append({'image_id':i,'category_id':1,'bbox':[x,y,x1-x,y1-y],'score':s})
    if not pred:return {'AP':0.,'AP50':0.,'AP75':0.}
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        api=COCO();api.dataset=gt;api.createIndex();d=api.loadRes(pred);ev=COCOeval(api,d,'bbox');ev.params.maxDets=[1,10,100];ev.evaluate();ev.accumulate();ev.summarize()
    return {'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2]),'AP_small_area':float(ev.stats[3]),'maxDets':100,'label_scope':'annotation_relative_not_certified_complete','stdout':log.getvalue()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--coco',action='store_true');p.add_argument('--thresholds',type=Path);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'railway_person_research'));plan=json.loads(a.plan.read_text(encoding='utf8'));rows=load_rows(root,plan);out=root/plan['output'];a.output.mkdir(parents=True,exist_ok=True)
    heldout=plan['purpose'].startswith(('test','external'));locked=json.loads(a.thresholds.read_text(encoding='utf8')) if a.thresholds else None
    if heldout and locked is None:raise RuntimeError('Formal test/external evaluation requires frozen validation thresholds')
    scores=sorted(set([.001,.005,.025,.075,.25,.5]+[round(float(x),2) for x in np.arange(.01,1,.01)]));results={};rng=np.random.default_rng(20261025)
    for spec in plan['variants']:
        name=spec['name'];folder=out/name;predictions={};times=[];calls=[]
        for row in rows:
            key=hashlib.sha256(row['image_id'].encode()).hexdigest()[:24];path=folder/(key+'.json')
            record=json.loads(path.read_text(encoding='utf8'))
            if record['image_id']!=row['image_id'] or record['source_sha256']!=row['image_sha256']:raise RuntimeError('Prediction image mismatch')
            boxes=np.asarray(record['predictions'],dtype=np.float64).reshape(-1,6)
            if not np.isfinite(boxes).all() or np.any(boxes[:,5]!=0):raise RuntimeError('Invalid boxes')
            predictions[row['image_id']]=boxes;times.append(record['analysis_seconds']);calls.append(len(record['actions']))
        thresholds={};per_primary=[]
        for score in scores:
            values=[counts(predictions[row['image_id']],row,score) for row in rows];total={k:sum(v[k] for v in values) for k in values[0]};thresholds[str(score)]=rates(total,len(rows))
            if score==.25:per_primary=values
        eligible=[(float(s),v) for s,v in thresholds.items() if v['UPI']<=.2];chosen=max(eligible,key=lambda x:(x[1]['matched'],-x[1]['UPI'],x[0])) if eligible and not heldout else None
        groups=sorted(set(r['capture_group_id'] for r in rows));group_stats=np.array([[sum(v[k] for r,v in zip(rows,per_primary) if r['capture_group_id']==g) for k in ['gt','matched','small_gt','small_matched']] for g in groups],dtype=np.float64);draws=group_stats[rng.integers(0,len(groups),size=(2000,len(groups)))].sum(axis=1)
        def ci(num,den):
            valid=draws[:,den]>0;return np.quantile(draws[valid,num]/draws[valid,den],[.025,.975]).tolist()
        operating=None
        if locked and name in locked['scores']:
            score=locked['scores'][name];operating={'score':score,'metrics':thresholds[str(score)],'selection_source':'frozen validation calibration; no test selection'}
        result={'images':len(rows),'variant':spec,'thresholds':thresholds,'validation_selection_UPI02':{'score':chosen[0],'metrics':chosen[1]} if chosen else None,'frozen_operating_point':operating,'primary_group_bootstrap':{'groups':len(groups),'replicates':2000,'seed':20261025,'recall_95_percentile':ci(1,0),'small_recall_95_percentile':ci(3,2),'scope':'descriptive; few correlated capture groups, not independent actors/sites'},'PC_times_from_execution_and_cache_not_a_latency_benchmark':{'mean':float(np.mean(times)),'P95':float(np.percentile(times,95))},'calls_per_image':{'min':min(calls),'max':max(calls)},'failed_images':0}
        if a.coco:result['annotation_relative_COCO']=coco_metrics(root,rows,predictions)
        results[name]=result;(a.output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps({'variant':name,'primary':thresholds['0.25'],'operating':result['validation_selection_UPI02']},ensure_ascii=False),flush=True)
        (a.output/(name+'_per_image_primary.json')).write_text(json.dumps([{'image_id':r['image_id'],'group':r['capture_group_id'],**v} for r,v in zip(rows,per_primary)],ensure_ascii=False),encoding='utf8')


if __name__=='__main__':main()
