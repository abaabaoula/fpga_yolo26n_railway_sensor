"""Exact ConvInteger CPU evaluator; board INT8/INT32 execution is unchanged."""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T

class ORTIntegerBackend:
    def __init__(self,shadow=False):
        self.shadow=shadow;self.sessions={};self.calls=0;self.checked_elements=0;self.bounds={}
    def execute(self,x,w,b,spec,constants):
        q=spec['quant'];sid=spec['id'];zx=int(constants[q['input_zp']]);wz=constants[q['weight_zp']].astype(np.int64).reshape(-1)
        if not np.all(wz==wz[0]):raise RuntimeError('ConvInteger requires a shared weight zero-point')
        if sid not in self.sessions:
            centered=w.astype(np.int64)-wz.reshape(-1,1,1,1)
            bound=255*np.abs(centered).reshape(w.shape[0],-1).sum(1)+np.abs(b.astype(np.int64));maximum=int(bound.max())
            if maximum>=2**31:raise RuntimeError('Integer accumulator overflow bound')
            self.bounds[sid]=maximum
            wu=np.bitwise_xor(w.view(np.uint8),np.uint8(128));k=spec['K'];pad=spec['padding'];stride=spec['stride']
            node=h.make_node('ConvInteger',['x','w','zx','zw'],['y'],strides=[stride,stride],pads=[pad]*4,group=spec['groups'],kernel_shape=[k,k])
            graph=h.make_graph([node],'exact_integer_conv',[h.make_tensor_value_info('x',T.UINT8,list(x.shape))],[h.make_tensor_value_info('y',T.INT32,None)],[nh.from_array(wu,'w'),nh.from_array(np.uint8(zx+128),'zx'),nh.from_array(np.uint8(int(wz[0])+128),'zw')])
            model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=8)
            so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1;so.enable_cpu_mem_arena=False
            so.add_session_config_entry('session.intra_op.allow_spinning','0');so.add_session_config_entry('session.inter_op.allow_spinning','0')
            self.sessions[sid]=ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])
        xu=np.bitwise_xor(x.view(np.uint8),np.uint8(128))
        value=self.sessions[sid].run(None,{'x':xu})[0][0]
        value+=b[:,None,None]
        if self.shadow:
            import torch
            import torch.nn.functional as F
            with torch.inference_mode():ref=F.conv2d(torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64).sub_(zx),torch.from_numpy(w.astype(np.float64))-float(wz[0]),torch.from_numpy(b.astype(np.float64)),stride=spec['stride'],padding=spec['padding'],groups=spec['groups']).numpy()[0]
            if not np.array_equal(value,ref):raise RuntimeError(('ORT integer accumulator mismatch',sid,int(np.count_nonzero(value!=ref))))
            self.checked_elements+=value.size
        self.calls+=1
        return value
