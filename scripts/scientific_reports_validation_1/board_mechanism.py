"""Same-firmware rectangular execution-factor experiment, after the joint campaign completes."""
import ctypes,fcntl,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import cv2,numpy as np
r=Path('/home/xilinx/jupyter_notebooks/scientific_reports_validation_1');base=r.parent/'yolo26_optimization_8'
prior=r.parent/'railway_person_optimization';rs=r.parent/'railway_system_optimization_1'
sys.path.insert(0,str(prior))
from railway_person_optimization.detector import RectDetector
out=r/'reports/mechanism_v1';out.mkdir(exist_ok=False)
contract=json.loads((r/'mechanism_reference_v1/manifest.json').read_text())
assert contract['formal_calls']==432
assert json.loads((r/'reports/campaign_v1/complete.json').read_text())['status']=='PASS'
pack=r/'board_pack_v1';images=json.loads((pack/'manifest.json').read_text())['images']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
library=rs/'builds/native_arm_v3/native_rs1.so'
assert sha(library)=='3ec0523f64a00f19f14330af562e840d2e890b7890c858e1f74943ec6d1a0d43'
memstat=Path('/sys/class/drm/card0/device/memstat_raw')
lock=(r.parent/'yolo26_stage_d/reports/accelerator_job.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'status':'RUNNING','phases':[],'programming':False,'formal_calls':0}
def save(): (out/'progress.json').write_text(json.dumps(state,indent=2))
def source(ref):
    s=images[ref['source_index']];a=cv2.imread(str(pack/s['file']))
    if ref['crop_xyxy']:
        x,y,x1,y1=ref['crop_xyxy'];a=np.ascontiguousarray(a[y:y1,x:x1])
    assert hashlib.sha256(a.tobytes()).hexdigest()==ref['decoded_input_sha256']
    return a
def engine(shape,options,resident):
    name='int8_384x640_railptq_v1' if shape=='384x640' else 'int8_256x1280_railptq_top384_v1'
    return RectDetector(base,prior/'candidates'/name,device='board',library=str(library),options=options,plan=resident)
try:
    configs=[{'shape':s,'options':o,'resident':p} for s in contract['profiles'] for o in contract['options'] for p in contract['resident']]
    for round_index in range(3):
        order=configs if round_index==0 else list(reversed(configs)) if round_index==1 else configs[4:]+configs[:4]
        for config in order:
            name=f"round{round_index}_{config['shape']}_flags{config['options']}_resident{int(config['resident'])}"
            state['phase']=name;save();refs=[x for x in contract['references'] if x['shape']==config['shape']]
            if round_index==1:refs.reverse()
            elif round_index==2:refs=refs[4:]+refs[:4]
            g=None;record={'config':config,'round':round_index,'samples':[],'warmups':[]}
            try:
                t=time.perf_counter();g=engine(**config);record['init_s']=time.perf_counter()-t
                for k in range(3):
                    a=g.infer(source(refs[0]),.25,return_raw=True)
                    record['warmups'].append({'ms':a['analysis_seconds']*1000,'cold':k==0})
                for ref in refs:
                    p=r/'mechanism_reference_v1'/ref['file'];assert sha(p)==ref['sha256']
                    expected=np.asarray(json.loads(p.read_text())['raw'],np.float32)
                    a=g.infer(source(ref),.25,return_raw=True)
                    assert np.array_equal(a['raw'],expected),(name,ref['image_id'])
                    hardware=g.graph.status();assert int(hardware['error'],0)==0 and hardware['version']=='0x80001'
                    record['samples'].append({'image_id':ref['image_id'],'group':ref['group'],'ms':a['analysis_seconds']*1000,
                        'native_timers':g.graph.timers.tolist(),'raw_exact':True,'hardware':hardware})
                    state['formal_calls']+=1;save()
                record['memory']=g.graph.memory();record['status']='PASS'
            except BaseException as e:
                record.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());raise
            finally:
                if g:g.close()
                record['closed_memstat']=memstat.read_text()
                (out/(name+'.json')).write_text(json.dumps(record,indent=2))
                assert int(memstat.read_text().split()[0])==0
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            state['phases'].append(name);save()
    # Independent diagnostics: sampled profiling does not enter the formal latency cohort.
    for shape in contract['profiles']:
        ref=next(x for x in contract['references'] if x['shape']==shape);a=source(ref)
        expected=np.asarray(json.loads((r/'mechanism_reference_v1'/ref['file']).read_text())['raw'],np.float32)
        for resident in contract['resident']:
            state['phase']=f'diagnostic_{shape}_resident{int(resident)}';save();g=None
            try:
                g=engine(shape,6276|32|64,resident);samples=[]
                for i in range(20):
                    value=g.infer(a,.25,return_raw=True);assert np.array_equal(value['raw'],expected)
                    detail,nodes=g.graph.detail()
                    if detail[14]:samples.append({'call':i+1,'ms':value['analysis_seconds']*1000,
                        'native_timers':g.graph.timers.tolist(),'detail':detail.tolist(),'nodes':nodes.tolist()})
                assert len(samples)==1 and samples[0]['call']==20
                (out/(state['phase']+'.json')).write_text(json.dumps({'status':'PASS','source':ref,'samples':samples,'timing_scope':'diagnostic_only'},indent=2))
            finally:
                if g:g.close()
                assert int(memstat.read_text().split()[0])==0
                ctypes.CDLL('libc.so.6').malloc_trim(0)
    assert state['formal_calls']==432
    state.update(status='PASS',phase='COMPLETE');save()
    (out/'complete.json').write_text(json.dumps(state,indent=2))
except BaseException as e:
    state.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps(state),flush=True)
