"""Analyze only fully completed raw campaigns; never substitute old24 results for new240."""
from common import *
import csv,importlib.util,os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output-subdir',required=True)
args=parser.parse_args()
assert Path(args.output_subdir).name==args.output_subdir and args.output_subdir not in ('.','..')
DEST=REPORT/args.output_subdir
DEST.mkdir(exist_ok=False)

import numpy as np
sys.path.insert(0,str(ROOT/'railway_person_research'))
spec=importlib.util.spec_from_file_location('metrics',ROOT/'scripts/railway_person_research/evaluate_matrix.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
b=REPORT/'board_results/reports'
assert json.loads((b/'campaign_v1/complete.json').read_text(encoding='utf8'))['status']=='PASS'
assert json.loads((b/'mechanism_v1/complete.json').read_text(encoding='utf8'))['status']=='PASS'
selected=[json.loads(s) for s in (REPORT/'joint240_rows.jsonl').read_text(encoding='utf8').splitlines()]
lookup={r['image_id']:r for r in selected};contract=json.loads((REPORT/'contract_v1.json').read_text(encoding='utf8'))
names=contract['joint']['configs'];results={};table=[];per_image=[];total_frames=0;exact=0
for name in names:
    samples=[];rounds=[];preds=[]
    for k in range(3):
        phase=b/'campaign_v1'/f'round{k}_{name}'
        record=json.loads((phase/'result.json').read_text(encoding='utf8'));assert record['status']=='PASS'
        rows=record['samples'];assert len(rows)==240 and {s['image_id'] for s in rows}==set(lookup)
        assert rows==[json.loads(s) for s in (phase/'samples.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(record['warmups'])==3 and int(record['closed_memstat'].split()[0])==0
        for s in rows:
            if name in ('global_opt','adaptive_opt'):
                assert s['check']['PC_exact'] and s['hardware']['version']=='0x80001' and int(s['hardware']['error'],0)==0
                exact+=1
        samples+=rows;rounds.append(float(np.mean([s['ms'] for s in rows])));preds.append({s['image_id']:s['predictions'] for s in rows})
    variation=[i for i in lookup if any(preds[j][i]!=preds[0][i] for j in (1,2))]
    bygroup={};counts=[]
    for row in selected:
        boxes=np.asarray(preds[0][row['image_id']],np.float32).reshape(-1,6)
        c=metrics.counts(boxes,row,.25);counts.append(c)
        times=[s['ms'] for s in samples if s['image_id']==row['image_id']]
        per_image.append({'method':name,'image_id':row['image_id'],'group':row['capture_group_id'],
            'mean_ms':float(np.mean(times)),**c})
    total={k:sum(c[k] for c in counts) for k in counts[0]};accuracy=metrics.rates(total,len(selected))
    for group in sorted({r['capture_group_id'] for r in selected}):
        values=[v for v in per_image if v['method']==name and v['group']==group]
        totals={k:sum(v[k] for v in values) for k in counts[0]}
        bygroup[group]={'images':len(values),'mean_ms':float(np.mean([v['mean_ms'] for v in values])),**metrics.rates(totals,len(values))}
    lat=np.asarray([s['ms'] for s in samples])
    result={'images':240,'formal_timed_frames':len(samples),'mean_ms':float(lat.mean()),'median_ms':float(np.median(lat)),
        'P95_ms':float(np.percentile(lat,95)),'max_ms':float(lat.max()),'round_means_ms':rounds,
        'accuracy_actual_ARM_round0':accuracy,'prediction_variation_images':variation,'per_group':bygroup}
    results[name]=result;total_frames+=len(samples)
    table.append({'method':name,'images':240,'timings':len(samples),'mean_ms':result['mean_ms'],'median_ms':result['median_ms'],
        'P95_ms':result['P95_ms'],'max_ms':result['max_ms'],'matched':total['matched'],'GT':total['gt'],
        'recall':accuracy['recall'],'small_recall_h_lt64':accuracy['small_recall'],'UPI':accuracy['UPI']})
assert total_frames==4320 and exact==1440
groups=sorted(results['adaptive_opt']['per_group']);rng=np.random.default_rng(20261103)
draws=rng.integers(0,len(groups),size=(2000,len(groups)));comparisons={}
for other in ('ncnn_global640','ncnn_adaptive','ncnn_int8_adaptive','ncnn_whole1280'):
    a=np.array([results[other]['per_group'][g]['mean_ms'] for g in groups])
    d=np.array([results['adaptive_opt']['per_group'][g]['mean_ms'] for g in groups])
    ratios=a[draws].mean(1)/d[draws].mean(1)
    comparisons[other]={'time_ratio_to_adaptive_PL':results[other]['mean_ms']/results['adaptive_opt']['mean_ms'],
        'group_bootstrap_descriptive_95_percentile':np.quantile(ratios,[.025,.975]).tolist(),
        'recall_difference_PL_minus_other':results['adaptive_opt']['accuracy_actual_ARM_round0']['recall']-results[other]['accuracy_actual_ARM_round0']['recall']}
summary={'status':'PASS','images':240,'groups':6,'formal_timed_frames':total_frames,'custom_PC_exact_checks':exact,
    'methods':results,'comparisons':comparisons,'partial_labels':True,'UPI_is_certified_FPR':False,
    'bootstrap':{'seed':20261103,'replicates':2000,'scope':'descriptive correlated capture groups, not independent sites or actors'},
    'data_role':'viewed validation, group-balanced metadata sample; not natural prevalence or unseen test'}
dump(DEST/'joint240_summary.json',summary)
folder=DEST/'tables';folder.mkdir(exist_ok=True)
for filename,values in [('joint240_actual_ARM.csv',table),('joint240_per_image.csv',per_image)]:
    with (folder/filename).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

mechanisms=[];formal=0
for p in sorted((b/'mechanism_v1').glob('round*.json')):
    v=json.loads(p.read_text(encoding='utf8'));assert v['status']=='PASS' and len(v['samples'])==12
    assert int(v['closed_memstat'].split()[0])==0
    for s in v['samples']:
        assert s['raw_exact'];formal+=1
        t=s['native_timers'];assert len(t)==16 and abs(t[7]-sum(t[i] for i in (0,1,2,5)))<1e-3
        mechanisms.append({**v['config'],'round':v['round'],'image_id':s['image_id'],'group':s['group'],'ms':s['ms'],
            'pack_ms':t[0],'DMA_wait_ms':t[1],'unpack_ms':t[2],'PL_busy_ms':t[3],'PL_compute_ms':t[4],
            'ARM_residual_including_norm_ms':t[5],'DW_subset_ms':t[6],'native_total_ms':t[7],
            'TX_payload_bytes':t[8],'RX_payload_bytes':t[9],'norm_subset_ms':t[10],
            'HP0_read_raw_counter':t[11],'HP0_write_raw_counter':t[12]})
assert formal==432
with (folder/'rectangular_mechanism_per_frame.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(mechanisms[0]));w.writeheader();w.writerows(mechanisms)
points=[]
for shape in ('384x640','256x1280'):
    for options in (4,4100,6276):
        for resident in (False,True):
            values=[s for s in mechanisms if s['shape']==shape and s['options']==options and s['resident']==resident]
            point={'shape':shape,'options':options,'resident':resident,'timings':len(values)}
            for k in list(mechanisms[0])[6:]:
                if k not in ('image_id','group') and isinstance(values[0].get(k),(float,int)):point[k]=float(np.mean([x[k] for x in values]))
            # Explicit fields avoid dependence on CSV/dict iteration for key costs.
            for k in ('ms','TX_payload_bytes','RX_payload_bytes'):point[k]=float(np.mean([x[k] for x in values]))
            points.append(point)
dump(DEST/'rectangular_mechanism_summary.json',{'status':'PASS','formal_calls':formal,'points':points,
    'raw_reference_exact_checks':formal,'diagnostics':4,'diagnostic_calls':80,
    'timing':'individual engine call; diagnostic sampling excluded','counter_scope':'native payload and raw HP0 values; no total CPU DDR claim',
    'subset_caution':'PL busy/compute are within DMA wait; DW and norm are within ARM residual, never add them again'})
print(json.dumps({'status':'PASS','joint_frames':total_frames,'mechanism_frames':formal,'custom_exact_joint':exact}))
