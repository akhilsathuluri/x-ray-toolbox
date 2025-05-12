import numpy as np
from pydrake.all import *
import sys
from joblib import Parallel, delayed
from pathlib import Path
sys.path.append(
    (Path(__file__).parent.parent.parent.parent.parent / "dsl4ras-pfd")
    .resolve()
    .__str__()
)
from SCARA_collision_free import ScaraSystem


class CollisionFree:
    def __init__(self):
        self.problem_name = "CollisionFree"
        self.problem_description = (
            "Problem to decompose collision free workspace of a scara robot"
        )
        self.plotter = np.array(
            [
                [0, 1],
                [2, 3],
                [4, 5],
            ]
        )
        num_robots = 3
        self.sim_system = ScaraSystem(num_robots=num_robots)
        nominalDV = np.array([0.8, 1.5, 1000, 0.9, 650, 160, 0.9, 10, 0.39, 0.26])
        dvs = np.tile(nominalDV, (num_robots, 1))
        self.sim_system.setup_robots_scene(dvs=dvs, modify_urdf=False)    

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        var_list = self.var.values.tolist()
        self.qoi_values = []
        for var in var_list:
            qoi = self.sim_system.compute_collision_in_config(var)
            self.qoi_values.append(qoi)
        self.qoi_values = np.array(self.qoi_values, dtype=int)
    
    def d_min(self):
        self.var["d_min"] = self.qoi_values