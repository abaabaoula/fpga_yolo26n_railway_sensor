"""Full-image fixed grid slicing reference; explicitly a SAHI-style adaptation."""
from railway_budget_research.policy import Config
from railway_budget_research.strategies import make_scheduler

class GridScheduler:
    def __init__(self):
        self.config = Config()
    def decide(self, shape, predictions, hint=None):
        height, width = shape[:2]
        tile_w, tile_h = max(1, round(width/2)), max(1, round(height/2))
        def starts(total, size):
            step = max(1, round(size*.8))
            return sorted(set(list(range(0,max(1,total-size+1),step))+[total-size]))
        crops = [[x,y,x+tile_w,y+tile_h] for y in starts(height,tile_h) for x in starts(width,tile_w)]
        return {"mode":"fixed_full_grid","crops":crops,"tile_fraction":.5,"overlap":.2,
                "implementation":"SAHI-style fixed slicing reference, not official SAHI software"}

def scheduler(name, protocol):
    return GridScheduler() if name=="grid_sahi" else make_scheduler(name, protocol)
