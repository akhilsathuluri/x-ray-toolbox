# %% Example SSO problem for CrashDesign

import numpy as np

# from problems.CrashDesign.library.CrashDesign import CrashDesign
from problems.DSL4RAS.library.DSL4RAS import DSL4RAS
import pandas as pd
import logging
from src.toolopt.XRayOpt import XRayOpt


# %%
sso = XRayOpt(log_level=logging.DEBUG)
sso.growth_rate = 0.08
sso.sample_size = 100
sso.max_exploration_iterations = 20
sso.max_consolidation_iterations = 20
sso.use_adaptive_growth_rate = True
sso._setup_problem(problem_path="problems/DSL4RAS", problem_class=DSL4RAS())
# sso._setup_problem(problem_path="problems/CrashDesign", problem_class=CrashDesign())
logging.basicConfig(level=logging.DEBUG)
# fmt: off

# shape is n_dv, 2
# sso._set_initial_box(np.array([[8.0e-01, 1.5e+00, 1.0e+03, 9.0e-01, 1.0e+02, 1.6e+02, 9.0e-01, 1.0e+01, 3.9e-01, 2.6e-01, 1.0e+02, 2.0e+01]] * 2).T)
sso._set_initial_box(np.vstack((sso.dv_l, sso.dv_u)).T)
# fmt: on
# sso._set_initial_box()
dv_box, box_measure = sso.run_sso_stochastic_iteration()
dv_sol_space = sso.export_optimisation_result(dv_box)


# %%
print(f"Final box measure: {box_measure}")
print(f"Final design variable box: {dv_box}")
# %%
