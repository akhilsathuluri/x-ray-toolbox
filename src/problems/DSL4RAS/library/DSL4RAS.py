import numpy as np
from pathlib import Path
import sys
from joblib import Parallel, delayed

sys.path.append("/home/wslaniakea/git/research/py-dsl4ras-sim-models")
from SCARA_example import get_accuracy_value


class DSL4RAS:
    def __init__(self):
        self.problem_name = "DSL4RAS"
        self.problem_description = "SCARA robot example for a pick and place problem"
        self.plotter = np.array(
            [
                [2, 0],
                [0, 4],
                [0, 5],
                [5, 1],
            ]
        )

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        var_list = self.var.values.tolist()

        self.accuracy_values = Parallel(n_jobs=-1)(
            delayed(get_accuracy_value)(var) for var in var_list
        )

    def accuracy(self):
        self.var["accuracy"] = self.accuracy_values
