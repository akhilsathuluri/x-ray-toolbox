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

# Initialise the tool class
xray = XRayViz()
# Generate a unique ID for one run of the tool
# We can use the sessionID to decide if we want to
# rerun the simulations or not
sessionID = -1
if 'sessionID' and 'prev_sessionID' not in st.session_state:
    st.session_state['sessionID'] = sessionID
    st.session_state['prev_sessionID'] = sessionID
    st.session_state.prev_sample_size = xray.sample_size

sessionID = np.random.randint(11, 99999)
st.session_state.sessionID = sessionID

# load the sliders
dv, qoi = xray.update_sliders()

# Initiate plots
figs = xray.initiate_plots()

# Plot the rectangles based on the sliders
rect_figs = xray.update_rectangles(dv, figs)

xray.export_options()
scatter_figs = xray.scatter_plots(rect_figs)
st.session_state.figs = scatter_figs

final_figs = xray.overlay_info(scatter_figs)
xray.plot_figs(final_figs)
xray.load_assets()

# Once one round of plotting is done, update the previous session ID
st.session_state.prev_sessionID = st.session_state.sessionID
st.session_state.prev_sample_size = xray.sample_size
