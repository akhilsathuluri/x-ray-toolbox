import numpy as np


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
        pass
