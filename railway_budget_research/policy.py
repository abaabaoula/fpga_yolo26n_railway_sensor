"""Image-only ROI scoring and prefix-bounded dynamic compute scheduling.

No label or row/sequence identifier is accepted by any inference decision API.
The budget is the number of fine calls, not estimated seconds. Latency must
be measured separately on the target device. Empty detections never imply Easy.
"""
from dataclasses import dataclass
import numpy as np
from railway_person_optimization.application import choose_strip

@dataclass(frozen=True)
class Config:
    uncertainty: bool=True
    coverage: bool=True
    geometry: bool=False
    focus_height: int=192
    mode: str='normal'
    mean_fine_budget: float=1.0
    easy_score: float=.85
    easy_min_height: float=128
    hard_marginal: float=.45

def geometry_hint(image):
    """Optional generic line convergence, never asserted to identify actual rails."""
    import cv2
    h,w=image.shape[:2];scale=320/w
    gray=cv2.cvtColor(cv2.resize(image,(320,max(1,round(h*scale)))),cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,60,160);edge[:round(.35*len(edge))]=0
    lines=cv2.HoughLinesP(edge,1,np.pi/180,25,minLineLength=40,maxLineGap=12)
    sides=[[],[]]
    if lines is not None:
        for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4):
            if abs(x1-x0)<8:continue
            m=(y1-y0)/(x1-x0)
            if .25<abs(m)<4:sides[int(m>0)].append((float(m),float(y0-m*x0),float(np.hypot(x1-x0,y1-y0))))
    candidates=[]
    for a in sorted(sides[0],key=lambda x:-x[2])[:6]:
        for b in sorted(sides[1],key=lambda x:-x[2])[:6]:
            x=(b[1]-a[1])/(a[0]-b[0]);y=a[0]*x+a[1]
            if 64<x<256 and .15*len(gray)<y<.7*len(gray):candidates.append((x,y))
    if len(candidates)<3:return None
    xy=np.median(candidates,axis=0)
    if np.median(np.linalg.norm(np.asarray(candidates)-xy,axis=1))>40:return None
    return [float(xy[0]/scale),float(xy[1]/scale)]

def features(predictions):
    p=np.asarray(predictions,dtype=np.float32).reshape(-1,6)
    p=p[(p[:,4]>=.01)&(p[:,5]==0)&(p[:,3]>p[:,1])&(p[:,2]>p[:,0])]
    return p

def propose(shape,predictions,config=Config(),hint=None):
    h,w=shape[:2];p=features(predictions);bh=p[:,3]-p[:,1]
    centers=(p[:,:2]+p[:,2:4])/2
    # Reliability prevents numerous floor-score background predictions from
    # dominating; uncertainty increases the value of unresolved candidates.
    confidence=p[:,4].astype(np.float64)
    weights=np.sqrt(confidence)*np.minimum(1,64/np.maximum(bh,1))
    if config.uncertainty:weights*=1-confidence
    crops=[choose_strip(shape,predictions,'top'),choose_strip(shape,predictions,'adaptive')]
    fh=min(h,config.focus_height);fw=min(w,5*fh)
    def focus(x,y):
        x=int(np.clip(round(x-fw/2),0,w-fw));y=int(np.clip(round(y-fh/2),0,h-fh));return (x,y,x+fw,y+fh)
    order=np.argsort(-weights,kind='stable')[:12]
    crops.extend(focus(*centers[i]) for i in order)
    if config.geometry and hint is not None:crops.append(focus(*hint))
    unique=list(dict.fromkeys(tuple(map(int,c)) for c in crops));rank=[]
    for crop in unique:
        x0,y0,x1,y1=crop;gain=min(256/(y1-y0),1280/(x1-x0))
        inside=(centers[:,0]>=x0)&(centers[:,0]<=x1)&(centers[:,1]>=y0)&(centers[:,1]<=y1)
        # Partial boxes and repeated coverage are discounted, not labelled as
        # negatives. This coverage is candidate support, not GT coverage.
        intersection=np.maximum(0,np.minimum(p[:,2],x1)-np.maximum(p[:,0],x0))*np.maximum(0,np.minimum(p[:,3],y1)-np.maximum(p[:,1],y0))
        fraction=intersection/np.maximum(1,(p[:,2]-p[:,0])*bh)
        support=inside.astype(float)*(fraction if config.coverage else 1)
        resolution=np.minimum(3,np.sqrt(gain/max(min(384/h,640/w),1e-9)))
        utility=float(np.sum(weights*support)*resolution)
        if config.geometry and hint is not None and x0<=hint[0]<=x1 and y0<=hint[1]<=y1:utility+=.1*float(weights.sum())
        rank.append({'crop':list(crop),'utility':utility,'support':support.tolist(),'gain':gain})
    if not len(p):return [rank[0]],{'eligible':0,'easy':False,'risk':'no person proposals: broad top fallback'}
    rank.sort(key=lambda r:(-r['utility'],-(r['crop'][2]-r['crop'][0]),r['crop'][1],r['crop'][0]))
    first=rank[0];covered=np.asarray(first['support'])
    for r in rank[1:]:
        marginal=np.asarray(r['support'])*(1-covered) if config.coverage else np.asarray(r['support'])
        r['marginal']=float(np.sum(weights*marginal)*min(3,np.sqrt(r['gain']/max(min(384/h,640/w),1e-9))))
    rest=sorted(rank[1:],key=lambda r:(-r['marginal'],r['crop'][1],r['crop'][0]))
    easy=bool(len(p) and np.all(confidence>=config.easy_score) and np.all(bh>=config.easy_min_height))
    return [first]+rest,{'eligible':len(p),'easy':easy,'weighted_first_coverage':float(np.sum(weights*covered)/max(weights.sum(),1e-9)),'marginal_ratio':rest[0]['marginal']/max(first['utility'],1e-9) if rest else 0.}

class Scheduler:
    def __init__(self,config=Config()):self.config=config;self.credit=0.;self.frames=0;self.fine_calls=0
    def decide(self,shape,predictions,hint=None):
        cfg=self.config;rank,diagnostic=propose(shape,predictions,cfg,hint)
        self.frames+=1;self.credit=min(2.,self.credit+cfg.mean_fine_budget)
        if cfg.mode=='global':count=0
        elif cfg.mode=='normal':count=1
        elif cfg.mode=='hard':count=min(2,len(rank))
        elif cfg.mode=='dynamic':
            count=0 if diagnostic['easy'] else 1
            if count and self.credit>=2 and diagnostic.get('marginal_ratio',0)>=cfg.hard_marginal:count=min(2,len(rank))
        else:raise ValueError(cfg.mode)
        if cfg.mode=='dynamic' and count>self.credit+1e-9:
            # At budgets >=1, Normal is always possible; lower budgets would
            # silently drop fallback coverage and are therefore rejected.
            raise RuntimeError('Insufficient budget for conservative fallback')
        if cfg.mean_fine_budget<1:raise ValueError('Conservative dynamic budget requires >=1 fine call per frame')
        self.credit-=count;self.fine_calls+=count
        return {'mode':['Easy','Normal','Hard'][count],'crops':[r['crop'] for r in rank[:count]],'ranking':rank,'diagnostic':diagnostic,'state':{'frames':self.frames,'fine_calls':self.fine_calls,'credit':self.credit}}
