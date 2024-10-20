import numpy as np
from pathlib import Path
import sys
from joblib import Parallel, delayed

sys.path.append("/home/wslaniakea/git/research/py-dsl4ras-sim-models")
from SCARA_bottom_up_mappings import ScaraSystem


class DSL4RAS:
    def __init__(self):
        self.problem_name = "DSL4RAS"
        self.problem_description = "SCARA robot example for a pick and place problem"
        self.plotter = np.array(
            # [[6, 8], [5, 9], [7, 5], [3, 7], [2, 9], [6, 2], [0, 5], [1, 8], [10, 11]]
            [[5, 9], [10, 11]]
        )
        # self.plotter = np.array([[ii, ii + 1] for ii in range(12 - 1)])
        self.system = ScaraSystem()
        self.system.meshcat_visualisation = False

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        var_list = self.var.values.tolist()

        self.qoi_values = np.array(
            Parallel(n_jobs=-1)(
                delayed(self.system.evaluate_qois)(var) for var in var_list
            )
        )

    def accuracy(self):
        self.var["accuracy"] = self.qoi_values[:, 0]

    def reach(self):
        self.var["reach"] = self.qoi_values[:, 1]

    def weight(self):
        self.var["weight"] = self.qoi_values[:, 2]

    def stabilisation_time(self):
        self.var["stabilisation_time"] = self.qoi_values[:, 3]
        # print(self.var["stabilisation_time"])
