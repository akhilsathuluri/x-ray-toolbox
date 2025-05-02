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
    seed = st.number_input(
        "Random seed", min_value=1, max_value=10000, value=42, step=1
    )
    growth_rate = st.number_input(
        "Growth rate", min_value=0.0, max_value=1.0, value=0.08, step=1e-2
    )
    sample_size = st.number_input(
        "Sample Size", min_value=1, max_value=10000, value=100, step=1
    )
    max_exploration_iterations = st.number_input(
        "Max exploration iterations",
        min_value=1,
        max_value=10000,
        value=20,
        step=1,
    )
    max_consolidation_iterations = st.number_input(
        "Max consolidation iterations",
        min_value=1,
        max_value=10000,
        value=20,
        step=1,
    )
    init_box_type = st.selectbox(
        "Initial box type",
        options=["domain", "midpoint", "random-bounds", "random-point"],
        index=0,
    )
    use_adaptive_growth_rate = st.checkbox(
        "Use adaptive growth rate", value=False, key="adaptive_growth_rate"
    )
    if st.button("Optimise"):
        sso = XRayOpt(seed=seed, log_level=logging.CRITICAL)
        # sso.growth_rate = 8e-2
        # sso.sample_size = 100
        sso.growth_rate = growth_rate
        sso.sample_size = sample_size
        sso.use_adaptive_growth_rate = use_adaptive_growth_rate
        sso.max_exploration_iterations = max_exploration_iterations
        sso.max_consolidation_iterations = max_consolidation_iterations
        sso._get_problem_info_from_app(xray)
        match init_box_type:
            case "domain":
                sso._set_initial_box(np.vstack((sso.dv_l, sso.dv_u)).T)
            case "midpoint":
                sso._set_initial_box()
            case "random-point":
                random_point = sso.rng.uniform(sso.dv_l, sso.dv_u)
                sso._set_initial_box(np.vstack((random_point, random_point)).T)
            case "random-bounds":
                random_lower_bound = sso.rng.uniform(sso.dv_l, sso.dv_u)
                random_upper_bound = sso.rng.uniform(random_lower_bound, sso.dv_u)
                sso._set_initial_box(
                    np.vstack(
                        (
                            random_lower_bound,
                            random_upper_bound,
                        )
                    ).T
                )
        sso.update_qoi_bounds(st.session_state["updated_qoi"])
        sso.use_adaptive_growth_rate = True
        with st.spinner("Running stochastic iteration optimisation...", show_time=True):
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
                "use_adaptive_growth_rate": bool(sso.use_adaptive_growth_rate),
                "init_box_type": str(init_box_type),
                "seed": int(seed),
            },
        }
        with open(xray.problem_path + "/output/optimisation_config.json", "w") as f:
            json.dump(plot_data, f, indent=4)

        st.success(
            "Data saved to " + xray.problem_path + "/output/optimisation_config.json"
        )

xray.export_options()
with st.spinner("Running simulations...", show_time=True):
    scatter_figs = xray.scatter_plots(rect_figs)
st.session_state.figs = scatter_figs

final_figs = xray.overlay_info(scatter_figs)
xray.plot_figs(final_figs)
xray.load_assets()

# Once one round of plotting is done, update the previous session ID
st.session_state.prev_sessionID = st.session_state.sessionID
st.session_state.prev_sample_size = xray.sample_size
