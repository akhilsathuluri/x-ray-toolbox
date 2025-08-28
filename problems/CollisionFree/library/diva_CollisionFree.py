import numpy as np
from pydrake.all import *
import sys
from pathlib import Path

sys.path.append(
    (Path(__file__).parent.parent.parent.parent.parent / "DIVA-R1.1.4-integrated")
    .resolve()
    .__str__()
)
from diva_collision_free import DIVAR114IntSystem


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
                [6, 7],
                [8, 9],
            ]
        )
        self.sim_system = DIVAR114IntSystem()
        self.sim_system.setup_robots_scene()

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
