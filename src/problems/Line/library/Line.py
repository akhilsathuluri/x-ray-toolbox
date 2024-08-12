import numpy as np
from joblib import Parallel, delayed

class Line:
    def __init__(self):
        self.problem_name = 'Line example'
        self.problem_description = 'Example problem demonstrating the usage of the XRay tool'
        self.plotter = np.array([[1,2],[2,0],[1,0]])
    
    def _compute_commons(self, dv_samples):
        self.var = dv_samples

    def L1(self):
        self.var['L1'] = self.var['x']-self.var['y']

    def L2(self):
        self.var['L2'] = self.var['z']-self.var['y']
