import numpy as np

class CrashDesign:
    def __init__(self):
        # self.var = dv_samples
        self.problem_name = 'Crash Design'
        self.problem_description = 'Example problem demonstrating the usage of the XRay tool'
        self.plotter = np.array([[2,0],[0,4],[0,5],[0,1],[2,4],[2,5],[2,1],[4,5],[4,1],[5,1]])

    def _compute_commons(self, dv_samples):
        self.var = dv_samples
        pass

    def a_max(self):
        self.var['a_max'] = self.var['F_2']/self.var['m']

    def E_rem(self):
        self.var['E_rem'] = self.var['m']/2*np.power(self.var['v_0'],2)-(self.var['F_1']*self.var['d_1c']+self.var['F_2']*self.var['d_2c'])

    def order(self):
        self.var['order'] = self.var['F_1']-self.var['F_2']
