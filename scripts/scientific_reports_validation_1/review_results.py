"""Outcome-neutral consistency and cost review of the completed real measurements."""
from common import *
import collections,csv
import numpy as np

b=REPORT/'board_results';campaign=b/'reports/campaign_v1';mechanism=b/'reports/mechanism_v1'
joint=json.loads((REPORT/'joint240_summary.json').read_text(encoding='utf8'))
points=json.loads((REPORT/'rectangular_mechanism_summary.json').read_text(encoding='utf8'))['points']
local_manifest=OUTPUT/'board_pack_v1/manifest.json'
c=json.loads((campaign/'contract.json').read_text(encoding='utf8'))
assert c['source_manifest_sha256']==sha(local_manifest)
assert c['runner_sha256']==sha(b/'scripts/board_campaign.py')==sha(ROOT/'scripts'/NAME/'board_campaign.py')
sources=[json.loads(s) for s in (REPORT/'joint240_rows.jsonl').read_text(encoding='utf8').splitlines()]
person_boxes=[o for r in sources for o in r['objects'] if o['class_name_raw']=='person']
assert len(person_boxes)==479 and sum(o['bbox_xyxy_raw'][3]-o['bbox_xyxy_raw'][1]<64 for o in person_boxes)==208
telemetry={};warmups=[];memories={};round_accuracy=[];latency_rows=[]
for name in joint['methods']:
    phases=[json.loads((campaign/f'round{k}_{name}/result.json').read_text(encoding='utf8')) for k in range(3)]
    samples=[s for p in phases for s in p['samples']]
    for p in phases:
        warmups.append({'method':name,'round':p['round'],'init_s':p['init_seconds'],
            'cold_ms':p['warmups'][0]['ms'],'warm_ms':[v['ms'] for v in p['warmups'][1:]]})
    process=[s[k]['process'] for s in samples for k in ('before','after')]
    def kb(field):return [float(p[field].split()[0]) for p in process if field in p]
    major=[s['after']['major_faults']-s['before']['major_faults'] for s in samples]
    minor=[s['after']['minor_faults']-s['before']['minor_faults'] for s in samples]
    telemetry[name]={'formal_frames':len(samples),'major_fault_delta_total':sum(major),'minor_fault_delta_total':sum(minor),
        'frames_with_major_fault':sum(v>0 for v in major),'VmRSS_observed_kB_range':[min(kb('VmRSS')),max(kb('VmRSS'))],
        'VmSwap_observed_kB_range':[min(kb('VmSwap')),max(kb('VmSwap'))],
        'frequency_sensor_values':sorted({str(v) for s in samples for t in ('before','after') for k,v in s[t]['sensors'].items() if 'freq' in k})}
    memories[name]=[p['memory'] for p in phases]
    for p in phases:
        assert int(p['closed_memstat'].split()[0])==0
        for s in p['samples']:
            latency_rows.append({'method':name,'round':p['round'],'image_id':s['image_id'],'group':s['group'],
                'ms':s['ms'],'calls':s['calls'],'major_fault_delta':s['after']['major_faults']-s['before']['major_faults']})
    for k,p in enumerate(phases):
        # Exact round output identity, already independently checked per source by analyze_joint_v2.
        assert not joint['methods'][name]['prediction_variation_images']
        round_accuracy.append({'method':name,'round':k,'same_predictions_as_round0':True})
assert len(latency_rows)==4320 and len(warmups)==18

def select(shape,options,resident):return next(p for p in points if p['shape']==shape and p['options']==options and p['resident']==resident)
costs=[]
for shape in ('384x640','256x1280'):
    candidate='int8_384x640_railptq_v1' if shape=='384x640' else 'int8_256x1280_railptq_top384_v1'
    plan=ROOT/'output/railway_person_optimization'/candidate/'compiled/resident_plan/resident_plan.json'
    p=json.loads(plan.read_text(encoding='utf8'))
    assert p['capacity_bytes']==131072
    counts=p['baseline_dense_DMA_calls'],p['resident_DMA_calls']
    for options in (4,4100,6276):
        off,on=select(shape,options,False),select(shape,options,True)
        round_pairs=[]
        for k in range(3):
            def mean(resident):
                path=mechanism/f'round{k}_{shape}_flags{options}_resident{int(resident)}.json'
                record=json.loads(path.read_text(encoding='utf8'))
                return float(np.mean([s['ms'] for s in record['samples']]))
            a,d=mean(False),mean(True);round_pairs.append({'round':k,'off_ms':a,'on_ms':d,'latency_reduction_percent':100*(1-d/a)})
        costs.append({'shape':shape,'options':options,'cache_pairs':len(p['pairs']),
            'baseline_DMA_calls':counts[0],'resident_DMA_calls':counts[1],
            'off_ms':off['ms'],'on_ms':on['ms'],'time_ratio':off['ms']/on['ms'],
            'latency_reduction_percent':100*(1-on['ms']/off['ms']),
            'TX_saved_bytes':int(off['TX_payload_bytes']-on['TX_payload_bytes']),
            'RX_saved_bytes':int(off['RX_payload_bytes']-on['RX_payload_bytes']),
            'HP0_read_saved_bytes':int(off['HP0_read_raw_counter']-on['HP0_read_raw_counter']),
            'HP0_write_saved_bytes':int(off['HP0_write_raw_counter']-on['HP0_write_raw_counter']),
            'all_rounds_faster':all(v['latency_reduction_percent']>0 for v in round_pairs),'rounds':round_pairs,
            'plan_sha256':sha(plan)})
diagnostics=[]
for path in sorted(mechanism.glob('diagnostic_*.json')):
    d=json.loads(path.read_text(encoding='utf8'));assert d['status']=='PASS' and len(d['samples'])==1
    s=d['samples'][0];assert s['call']==20 and s['detail'][14]==1
    assert np.isfinite(s['detail']).all() and np.isfinite(s['native_timers']).all()
    diagnostics.append({'file':path.name,'sampled_call':20,'sampled_ms':s['ms'],'samples':1,'scope':'profile-only, excluded from formal measurements'})
assert len(diagnostics)==4
rtl_paths=['yolo26_optimization_8/hardware/rtl/axi_traffic_monitor.sv',
    'yolo26_optimization_8/hardware/rtl/config_registers.sv',
    'railway_system_optimization_1/native_sources/candidate_v3/graph_executor.cpp']
rtl={p:sha(ROOT/p) for p in rtl_paths}
monitor=(ROOT/rtl_paths[0]).read_text(encoding='utf8')
assert "read_bytes+32'd8" in monitor and 'write_strobe[i]' in monitor
registers=(ROOT/rtl_paths[1]).read_text(encoding='utf8')
assert "araddr==12'h180)rdata<=ddr_read_bytes" in registers and "araddr==12'h184)rdata<=ddr_write_bytes" in registers

rng=np.random.default_rng(20261103);groups=sorted(joint['methods']['adaptive_opt']['per_group'])
draws=rng.integers(0,6,size=(2000,6));recall_ci={}
for method,result in joint['methods'].items():
    gp=np.array([[result['per_group'][g][k] for k in ('gt','matched','small_gt','small_matched')] for g in groups])
    totals=gp[draws].sum(axis=1);valid=totals[:,0]>0;small=totals[:,2]>0
    recall_ci[method]={'recall_descriptive_95':np.quantile(totals[valid,1]/totals[valid,0],[.025,.975]).tolist(),
        'small_recall_descriptive_95':np.quantile(totals[small,3]/totals[small,2],[.025,.975]).tolist(),
        'scope':'Six correlated acquisition groups; not an independent-population confidence claim'}
review={'status':'PASS_WITH_SCIENTIFIC_LIMITS','raw_integrity':'PASS','main_samples':4320,'custom_prediction_checks':1440,
    'mechanism_raw_checks':432,'diagnostic_raw_checks':80,'separately_recorded_main_warmups':54,
    'separately_recorded_mechanism_warmups':108,'diagnostic_only_calls':80,
    'predictions_identical_across_rounds':True,'telemetry':telemetry,'warmups':warmups,'native_memory':memories,
    'cache_effects':costs,'diagnostics':diagnostics,'recall_intervals':recall_ci,
    'HP0_counter_units':'bytes: read handshake increments8, write handshake adds enabled8-bitWSTRB bytes; addresses0x180/0x184 match native accumulators',
    'counter_source_sha256':rtl,'counter_scope':'PL HP0 observation only, not all CPU DDR traffic or physical DRAM energy',
    'scientific_limits':['6correlated viewed-validation groups, not unseen sites','only3positive external groups',
        'partial railway labels; UPI not verified FPR','major-fault/long-tail observations retained; no causal attribution without control',
        'power unmeasured','numerical correctness against frozen quantized reference, not lossless PTQ relative to FP32',
        'same-v3 flags are cumulative, not independent new modules','no claim of optimal ARM software or universal accelerator superiority']}
dump(REPORT/'result_review.json',review)
tables=REPORT/'tables'
with (tables/'joint240_all_timed_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(latency_rows[0]));w.writeheader();w.writerows(latency_rows)
with (tables/'cache_effects.csv').open('w',newline='',encoding='utf-8-sig') as f:
    simple=[{k:v for k,v in c.items() if k!='rounds'} for c in costs]
    w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
print(json.dumps({'status':review['status'],'cache':[{k:c[k] for k in ('shape','options','latency_reduction_percent','cache_pairs','baseline_DMA_calls','resident_DMA_calls','all_rounds_faster')} for c in costs],
    'major_fault_frames':{m:v['frames_with_major_fault'] for m,v in telemetry.items()}}))
