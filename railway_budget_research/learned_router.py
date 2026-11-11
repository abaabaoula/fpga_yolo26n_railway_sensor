"""Small CPU ridge utility router. Frozen detector; always one fine call.

Trained only on supplied-label utility, a partial-annotation lower-bound target.
Never interprets missing annotations as detector background/negative boxes.
"""
import numpy as np
from .policy import features
CHOICES=['old_adaptive','fixed_focus','fixed_focus128','roi_score']
def vector(shape,predictions):
    h,w=shape[:2];p=features(predictions);n=len(p)
    if not n:return np.zeros(30,np.float64)
    c=p[:,4];height=p[:,3]-p[:,1];x=(p[:,0]+p[:,2])/(2*w);y=(p[:,1]+p[:,3])/(2*h)
    v=[np.log1p(n),float(c.max()),float(c.mean()),float(c.std()),float(np.mean(height/h)),float(np.std(height/h)),float(x.mean()),float(x.std()),float(y.mean()),float(y.std())]
    v.extend(np.log1p([np.sum((height>=lo)&(height<hi)) for lo,hi in [(0,32),(32,64),(64,128),(128,100000)]]))
    v.extend(np.log1p([np.sum((c>=lo)&(c<hi)) for lo,hi in [(.01,.05),(.05,.25),(.25,.5),(.5,1.01)]]))
    v.extend(np.histogram(x,bins=[0,.25,.5,.75,1],weights=np.sqrt(c))[0]/max(np.sqrt(c).sum(),1e-9))
    v.extend(np.histogram(y,bins=[0,.25,.5,.75,1],weights=np.sqrt(c))[0]/max(np.sqrt(c).sum(),1e-9))
    v.extend([float(np.mean(height<64)),float(np.mean(c*(1-c))),float(np.mean((x>.25)&(x<.75)&(y>.15)&(y<.35))),float(w/h)])
    out=np.asarray(v,np.float64);assert out.shape==(30,) and np.isfinite(out).all();return out
def fit(x,y,alpha=10.):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1;z=(x-mean)/scale
    design=np.c_[np.ones(len(z)),z];penalty=np.eye(design.shape[1])*alpha;penalty[0,0]=0
    beta=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {'choices':CHOICES,'mean':mean.tolist(),'scale':scale.tolist(),'beta':beta.tolist(),'ridge_alpha':alpha,'feature_count':30,'scope':'supplied-instance recovery utility, partial annotations; not object detector training or background classification'}
def predict(model,shape,coarse):
    v=vector(shape,coarse);z=(v-np.array(model['mean']))/np.array(model['scale']);utilities=np.r_[1,z]@np.asarray(model['beta']);return model['choices'][int(np.argmax(utilities))],utilities.tolist()
