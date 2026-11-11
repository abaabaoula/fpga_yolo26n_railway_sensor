"""Unlabelled development-only calibration, no training or annotation changes."""
import argparse,os,shutil,sys,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--shape',required=True);p.add_argument('--crop-height',type=int,default=0);p.add_argument('--method',choices=['minmax','percentile'],default='minmax');p.add_argument('--output',type=Path);p.add_argument('--source',type=Path);p.add_argument('--dataset-root',type=Path,default=DATA);a=p.parse_args();hw=tuple(map(int,a.shape.split('x')))
suffix='_top'+str(a.crop_height) if a.crop_height else ''
method_suffix='_pct9999_stream' if a.method=='percentile' else ''
out=a.output or OUTPUT/('int8_'+a.shape+'_railptq'+suffix+method_suffix+'_v1');out.mkdir(parents=True,exist_ok=False)
os.environ.update(OMP_NUM_THREADS='2',YOLO_CONFIG_DIR=str(out/'ultralytics'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch,onnx
from onnxruntime.quantization import quantize_static,QuantFormat,QuantType,CalibrationMethod,CalibrationDataReader
from railway_person_optimization.geometry import preprocess
from compile_model import compile_rect,census
torch.set_num_threads(2);cv2.setNumThreads(2)
source=a.source or OUTPUT/('int8_'+a.shape+'_frozen_v1')/'official_source.onnx'
assert source.is_file(),source
fp=out/'official_source.onnx';shutil.copy2(source,fp)
excluded=set(json.loads((PROTOCOL/'development_screen_384.json').read_text()))
groups=collections.defaultdict(list)
for r in rows('development'):
    if r['image_id'] not in excluded:groups[r['sequence_id']].append(r)
for seq in groups:groups[seq].sort(key=lambda r:hashlib.sha256(('railcalib-v1:'+r['image_id']).encode()).hexdigest())
chosen=[]
for i in range(128):
    seq=sorted(groups)[i%len(groups)];chosen.append(groups[seq][i//len(groups)])
assert len({r['image_id'] for r in chosen})==128
assert not ({r['image_id'] for r in chosen}&excluded)
dump(out/'calibration_ids.json',[{'image_id':r['image_id'],'image_sha256':r['image_sha256']} for r in chosen])
class Reader(CalibrationDataReader):
    def __init__(self):self.it=iter(chosen);self.count=0
    def get_next(self):
        try:r=next(self.it)
        except StopIteration:return None
        path=a.dataset_root/r['image_id'];assert sha(path)==r['image_sha256']
        self.count+=1
        if self.count%16==0:print('UNLABELLED_DEVELOPMENT_CALIBRATION',self.count,flush=True)
        image=cv2.imread(str(path))
        if a.crop_height:image=image[:a.crop_height]
        return {'images':preprocess(image,hw)[0]}
qdq=out/'allconv_int8.onnx';reader=Reader()
extra={'ActivationSymmetric':False,'WeightSymmetric':True,'CalibMaxIntermediateOutputs':15}
if a.method=='percentile':
    extra['CalibPercentile']=99.99
    # Stock HistogramCalibrater retains every raw intermediate until EOF.
    # Keep the same ORT running histogram, feeding one sample per collection
    # call. This is an explicit streaming calibration candidate, not claimed
    # byte-identical to a single all-images histogram with fixed bin edges.
    from onnxruntime.quantization.calibrate import HistogramCalibrater
    original_collect=HistogramCalibrater.collect_data
    def bounded_collect(calibrator,data_reader):
        class One:
            def __init__(self,value):self.value=value
            def get_next(self):value=self.value;self.value=None;return value
        while True:
            value=data_reader.get_next()
            if value is None:break
            original_collect(calibrator,One(value))
    HistogramCalibrater.collect_data=bounded_collect
quantize_static(str(fp),str(qdq),reader,quant_format=QuantFormat.QDQ,op_types_to_quantize=['Conv'],per_channel=True,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,calibrate_method=CalibrationMethod.Percentile if a.method=='percentile' else CalibrationMethod.MinMax,extra_options=extra)
assert reader.count==128
dw,program,plan=compile_rect(qdq,out,hw)
record={'status':'COMPILED','shape_hw':hw,'actual_input':[1,3,*hw],'source_weights_sha256':sha(BASE/'reproduction_inputs/yolo26n.pt'),'training':False,'recalibration':True,'calibration_method':a.method,'percentile':99.99 if a.method=='percentile' else None,'calibration_images':128,'calibration_top_crop_height':a.crop_height,'calibration_split':'development only, disjoint from development_screen384 and all validation/test','labels_used_for_calibration':False,'integer_depthwise':dw,'program_sha256':sha(program/'graph.bin'),'resident_pairs':len(plan['pairs']),'resident_DMA_calls':plan['resident_DMA_calls']}
dump(out/'preparation.json',record);dump(out/'capacity.json',census(out));print(json.dumps(record),flush=True)
