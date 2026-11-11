"""Fixed and prior adaptive baselines with the same decision interface."""
from .policy import Config,Scheduler
from railway_person_optimization.application import choose_strip
class BaselineScheduler:
    def __init__(self,name):self.name=name;self.config=Config();self.frames=0
    def decide(self,shape,predictions,hint=None):
        h,w=shape[:2];self.frames+=1
        if self.name=='global':crops=[]
        elif self.name in ['fixed_top','old_adaptive']:crops=[list(choose_strip(shape,predictions,'top' if self.name=='fixed_top' else 'adaptive'))]
        elif self.name in ['fixed_focus','fixed_focus128']:
            fh=min(h,128 if self.name=='fixed_focus128' else 192);fw=min(w,fh*5);x=round((w-fw)/2);y=max(0,min(h-fh,round(h*.25-fh/2)));crops=[[x,y,x+fw,y+fh]]
        else:raise ValueError(self.name)
        return {'mode':'Normal' if crops else 'Easy','crops':crops,'baseline':self.name,'frames':self.frames}
class LearnedScheduler:
    def __init__(self,model,protocol):self.model=model;self.protocol=protocol;self.config=Config();self.frames=0
    def decide(self,shape,predictions,hint=None):
        from .learned_router import predict
        name,utility=predict(self.model,shape,predictions);self.frames+=1
        decision=make_scheduler(name,self.protocol).decide(shape,predictions,hint)
        return {**decision,'learned_router_choice':name,'learned_utilities':utility,'learned_frames':self.frames,'research_only':'partial-label utility model; not safety risk prediction'}
def make_scheduler(name,protocol,model=None):
    if name=='learned_router':
        if model is None:raise ValueError('Frozen learned_router.json is required')
        return LearnedScheduler(model,protocol)
    return Scheduler(Config(**protocol['methods'][name])) if name in protocol['methods'] else BaselineScheduler(name)
