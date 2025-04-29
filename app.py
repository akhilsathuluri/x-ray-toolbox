# This code is a python version of the X-Ray tool for finding solution spaces
# of multi-dimensional design problems.

# X-Ray Tool v0.0.3a
# Copyright (C) 2021, Akhil Sathuluri
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import numpy as np
from src.description import *
from src.toolviz.XRayViz import XRayViz
from src.toolopt.XRayOpt import XRayOpt
import logging
import json
import os

# Initialise the tool class
xray = XRayViz()
# Generate a unique ID for one run of the tool
# We can use the sessionID to decide if we want to
# rerun the simulations or not
sessionID = -1
if "sessionID" and "prev_sessionID" not in st.session_state:
    st.session_state["sessionID"] = sessionID
    st.session_state["prev_sessionID"] = sessionID
    st.session_state.prev_sample_size = xray.sample_size

sessionID = np.random.randint(11, 99999)
st.session_state.sessionID = sessionID

# load the sliders
dv, qoi = xray.update_sliders()

# Initiate plots
figs = xray.initiate_plots()

# Plot the rectangles based on the sliders
rect_figs = xray.update_rectangles(dv, figs)

# add option to optimise
# xray.optimisation_options()
run_opt = st.sidebar.expander("Optimisation")
with run_opt:
    st.write("Runs stochastic iteration optimisation")
    if st.button("Optimise"):
        sso = XRayOpt(log_level=logging.DEBUG)
        sso.growth_rate = 8e-2
        sso.sample_size = 100
        sso._get_problem_info_from_app(xray)
        sso._set_initial_box()
        sso.update_qoi_bounds(st.session_state["updated_qoi"])
        dv_box, box_measure = sso.run_sso_stochastic_iteration()
        dv_sol_space = sso.export_optimisation_result(dv_box)
        st.session_state["updated_dv"] = dv_sol_space
        result_outpath = xray.problem_path + "/output"
        os.makedirs(result_outpath, exist_ok=True)
        plot_data = {
            "box_measure": float(box_measure),
            "optimization_params": {
                "growth_rate": float(sso.growth_rate),
                "sample_size": int(sso.sample_size),
                "max_exploration_iterations": int(sso.max_exploration_iterations),
                "max_consolidation_iterations": int(sso.max_consolidation_iterations),
            },
        }
        with open(xray.problem_path + "/output/optimisation_config.json", "w") as f:
            json.dump(plot_data, f, indent=4)

        st.success(
            "Data saved to " + xray.problem_path + "/output/optimisation_config.json"
        )

xray.export_options()
scatter_figs = xray.scatter_plots(rect_figs)
st.session_state.figs = scatter_figs

final_figs = xray.overlay_info(scatter_figs)
xray.plot_figs(final_figs)
xray.load_assets()

# Once one round of plotting is done, update the previous session ID
st.session_state.prev_sessionID = st.session_state.sessionID
st.session_state.prev_sample_size = xray.sample_size
