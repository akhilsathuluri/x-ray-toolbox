import numpy as np
from joblib import Parallel, delayed
# from utils import *
import tqdm

def _compute_bottom_up_mappings(dv):
    print(dv)
    L1 = dv[0]-dv[1]
    L2 = dv[2]-dv[1]
    return L1, L2

class Line:
    def __init__(self):
        self.problem_name = 'Line example'
        self.problem_description = 'Example problem demonstrating the usage of the XRay tool'
        self.plotter = np.array([[1,2],[2,0],[1,0],[0,1,2]])
        # Also define information between variables to help generate ADG
        # self.bottom_up_mapping_list = np.array([[['x', 'y'], ['L1']], [['y', 'z'], ['L2']]])

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        var_list = self.var.values.tolist()

        # with tqdm_joblib(tqdm(total=len(var_list))) as progress_bar:
        with (tqdm(total=len(var_list))) as progress_bar:
            self.results = Parallel(n_jobs=-1,backend='loky')(delayed(_compute_bottom_up_mappings)(var_list[i]) for i in range(len(var_list)))
        self.results_array = np.array(self.results)
        pass

    def L1(self):
        self.var['L1'] = self.results_array[:, 0]

    def L2(self):
        self.var['L2'] = self.results_array[:, 1]
