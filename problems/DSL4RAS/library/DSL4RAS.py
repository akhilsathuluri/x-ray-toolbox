import numpy as np
import sys
from joblib import Parallel, delayed
from pathlib import Path

sys.path.append(
    (Path(__file__).parent.parent.parent.parent.parent / "dsl4ras-pfd")
    .resolve()
    .__str__()
)
from SCARA_bottom_up_mappings import ScaraSystem


class DSL4RAS:
    def __init__(self):
        self.problem_name = "DSL4RAS"
        self.problem_description = "SCARA robot example for a pick and place problem"
        self.plotter = np.array(
            [
                [0, 5],
                [1, 8],
                [2, 9],
                [3, 7],
                [5, 9],
                [6, 2],
                [6, 8],
                [7, 5],
                [10,11],
            ]
        )
        self.sim_system = ScaraSystem()

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        var_list = self.var.values.tolist()
        self.qoi_values = np.array(
            Parallel(n_jobs=-1)(
                # delayed(self.sim_system.evaluate_qois)(var) for var in var_list
                delayed(self.sim_system.evaluate_qois_MPC)(var)
                for var in var_list
            )
        )

    def accuracy(self):
        self.var["accuracy"] = self.qoi_values[:, 0]

    def reach(self):
        self.var["reach"] = self.qoi_values[:, 1]

    def robot_weight(self):
        self.var["robot_weight"] = self.qoi_values[:, 2]

    def pose_stabilisation(self):
        self.var["pose_stabilisation"] = self.qoi_values[:, 3]
