import numpy as np
from pathlib import Path
import pandas as pd

# This is a standalone class that can be used just for solution space based optimisation independant of the x-ray visualisation tool
# I want to add
#   - Stochastic iteration
#   - Fix rewriting the same code for example for sampling
#   - Add interface to the XRayViz but keep it decoupled
#   - Parallelisation support for quick optimisation


# what you get -- dv bounds, qoi bounds, initial guess of a good design, num_samples, num_iters?,
# what comes out -- dv_bounds_premissible


class XRayOpt:
    def __init__(self):
        self.problem_path = Path("problems/CrashDesign")
        self.problem_name = "Crash Design"
        dv = pd.read_csv(self.problem_path + "/input/dv_space.csv", dtype=str)
        qoi = pd.read_csv(self.problem_path + "/input/qoi_space.csv", dtype=str)
        dv.iloc[:, 1:3] = dv.iloc[:, 1:3].astype(np.float64)
        qoi.iloc[:, 1:3] = qoi.iloc[:, 1:3].astype(np.float64)
        dv_size = dv.shape[0]
        qoi_size = qoi.shape[0]

        self.problem_dv = dv
        self.problem_qoi = qoi
        self.problem_dv_size = dv_size
        self.problem_qoi_size = qoi_size

        pass
