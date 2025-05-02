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
import pandas as pd
import os
import plotly.graph_objs as go
from src.description import *
from importlib import import_module
import plotly.io as pio
import json
import itertools
import seaborn as sb


# Main class containing the features of the tool
class XRayViz:
    def __init__(self):
        # ############################################################################
        # ######################## Page configuration ################################
        # ############################################################################
        # Declare application name and version
        version = "v0.1a"
        name = "X-Ray Tool G7"
        st.set_page_config(
            layout="wide", page_title=name + version, page_icon=":flashlight:"
        )
        # Setting application details
        st.sidebar.title(name + version)

        # Load instructions of usage
        info_expander = st.expander("Getting started")
        with info_expander:
            st.markdown(help_text)

        # ############################################################################
        # ######################## Defaults setup ####################################
        # ############################################################################
        # Set problem seed
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        # Step movement for the design variable sliders
        self.step_disc = 500
        self.sample_size_step = 1
        self.sample_size_min = 1

        # Size of each plot
        self.plotsx = 450
        self.plotsy = 280

        # Setting margins of the plots
        self.plot_margin = dict(l=25, r=25, b=0, t=30)
        self.plot_font = dict(family="Computer Modern", size=12, color="Black")

        ############################################################################
        ######################## Problem dropdown ##################################
        ############################################################################
        # A dropdown to list all the available problems
        problem_config = st.sidebar.expander("Problem configuration")
        with problem_config:
            problem_list = ["-"] + os.listdir("./problems/")
            self.selected_problem = problem_config.selectbox(
                "Select problem", problem_list
            )
            if self.selected_problem == "-":
                st.stop()
            # Problem path
            self.problem_path = r"./problems/" + self.selected_problem
            self.sol_sample_size = self.sample_size_min
            # Set sample size
            self.sample_size = problem_config.number_input(
                "Sample size",
                min_value=self.sample_size_min,
                step=self.sample_size_step,
                value=self.sol_sample_size,
            )
            self.is_pfd = problem_config.checkbox(
                "PFD", value=False, key="is_pfd"
            )
            if self.is_pfd:
                st.success("Absolutely nothing will happen")
                st.stop()
            self.rerun_button = st.sidebar.button("Rerun")

        ############################################################################
        ######################## Load problem variables ############################
        ############################################################################
        # Reading initial input variable space from a csv file
        dv = pd.read_csv(self.problem_path + "/input/dv_space.csv", dtype=str)
        qoi = pd.read_csv(self.problem_path + "/input/qoi_space.csv", dtype=str)
        # Converting the data into float
        dv.iloc[:, 1:3] = dv.iloc[:, 1:3].astype(np.float64)
        qoi.iloc[:, 1:3] = qoi.iloc[:, 1:3].astype(np.float64)

        # Save problem level details that do not change
        self.problem_dv = dv
        self.problem_qoi = qoi
        self.problem_dv_size = dv.shape[0]
        self.problem_qoi_size = qoi.shape[0]

        # Initiate session state with default values
        if "updated_dv" and "updated_qoi" not in st.session_state:
            st.session_state["updated_dv"] = self.problem_dv
            st.session_state["updated_qoi"] = self.problem_qoi
            # Save the previous state of the sliders too
            st.session_state["prev_updated_dv"] = self.problem_dv

        ############################################################################
        ######################## Load problem class ################################
        ############################################################################
        mod = import_module(
            "problems." + self.selected_problem + ".library." + self.selected_problem
        )
        prob_class = getattr(mod, self.selected_problem)
        # Load the problem class and initiate an instance
        self.prob = prob_class()
        # Add the title of the problem
        st.title(self.prob.problem_name)
        # Add problem description
        with st.expander("Description"):
            st.write(self.prob.problem_description)
        # get the functions from the problem
        self.prob.eval_list = self.problem_qoi.Variables

        ############################################################################
        ######################## Plot configuration ################################
        ############################################################################
        plot_config = st.sidebar.expander("Plot configuration")
        with plot_config:
            # Adding a slider to manipulate the opacity of the constraints
            self.qoi_alpha = plot_config.slider(
                "QoI opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.15
            )
            # Select the number of plots in a given row
            self.plot_cols = int(
                plot_config.number_input("Columns", min_value=3, step=1)
            )
            self.marker_size = plot_config.slider(
                "Marker size", min_value=3.0, max_value=12.0, value=7.5, step=0.5
            )
            self.scatter_markers_ok = dict(
                color="rgba(10,255,10,1)", size=self.marker_size
            )

            # Generates plot between all DVs
            self.plot_matrix = st.checkbox("Generate variable matrix plots")
            if self.plot_matrix:
                # Generate only the required plots
                index_list = list(range(len(self.problem_dv.Variables)))
                self.plotter = np.array(list(itertools.combinations(index_list, 2)))

            # Remove label dropdown if not necessary
            self.label_dropdown = st.checkbox("Activate label description", value=False)

            # Load the button only if there are saved files
            self.load_sol = st.checkbox("Load solution", value=False)

            if self.load_sol:
                # Check if saved solutions are present
                if os.path.isfile(self.problem_path + "/output/dv_solution_space.csv"):
                    # Load solution spaces
                    self.solution_dv = pd.read_csv(
                        self.problem_path + "/output/dv_solution_space.csv"
                    )
                    self.solution_qoi = pd.read_csv(
                        self.problem_path + "/output/qoi_solution_space.csv"
                    )
                    # Load plot_data
                    with open(
                        self.problem_path + "/output/" + "plot_data.json", "r"
                    ) as f:
                        plot_data = json.load(f)
                    # Update the sample size from the saved data
                    self.sample_size = plot_data["sample_size"]
                    st.success("Saved solution loaded successfully")
                    st.session_state.updated_dv

                else:
                    st.error(
                        "Saved solution not available, please save solution spaces first"
                    )
                    st.stop()

            ############################################################################
            ######################## Problem probe #####################################
            ############################################################################
            self.collapse_space = plot_config.radio(
                "Collapse design space", ("None", "Nominal")
            )

    def update_sliders(self):
        with st.sidebar.form("dv_form"):
            variable_container = st.container()
            ############################################################################
            ######################## Variable sliders ##################################
            ############################################################################
            dv_expander = variable_container.expander("Design variables")
            # Do not reset sliders but use the session state slider values
            slider_dv = st.session_state["updated_dv"]
            slider_qoi = st.session_state["updated_qoi"]

            # However if either load_sol or collapse can change this
            if self.load_sol:
                slider_dv = self.solution_dv
                slider_qoi = self.solution_qoi
                st.session_state["updated_dv"] = self.solution_dv
                st.session_state["updated_qoi"] = self.solution_qoi

            if self.collapse_space == "Nominal":
                # Set the DVs to be a nominal design
                temp_state = st.session_state["updated_dv"]
                nominal_dv = (temp_state["Lower"] + temp_state["Upper"]) / 2
                # nominal_dv = st.session_state['updated_dv'].mean(axis=1)
                st.session_state["updated_dv"].Lower = nominal_dv
                st.session_state["updated_dv"].Upper = nominal_dv
                slider_dv = st.session_state["updated_dv"]

            with dv_expander:
                # Capture the slider step for collapse function
                slider_steps = []
                # Make sliders available in the sidebar
                for i in range(self.problem_dv_size):
                    slider_step = (
                        self.problem_dv.Upper[i] - self.problem_dv.Lower[i]
                    ) / self.step_disc
                    slider_steps.append(slider_step)
                    (slider_dv.loc[i, "Lower"], slider_dv.loc[i, "Upper"]) = (
                        dv_expander.slider(
                            self.problem_dv.Variables[i],
                            min_value=self.problem_dv.Lower[i],
                            max_value=self.problem_dv.Upper[i],
                            value=(
                                slider_dv.Lower[i],
                                slider_dv.Upper[i],
                            ),
                            step=slider_step,
                            key="dv_slider" + str(i),
                            format=f"%0.{int(np.abs(np.log10(slider_step)))}f"
                        )
                    )

            qoi_expander = variable_container.expander("Quantities of Interest")
            with qoi_expander:
                # Make sliders available in the sidebar
                for i in range(self.problem_qoi_size):
                    slider_step = (self.problem_qoi.Upper[i] - self.problem_qoi.Lower[i]) / self.step_disc
                    (slider_qoi.loc[i, "Lower"], slider_qoi.loc[i, "Upper"]) = (
                        qoi_expander.slider(
                            self.problem_qoi.Variables[i],
                            min_value=self.problem_qoi.Lower[i],
                            max_value=self.problem_qoi.Upper[i],
                            value=(
                                slider_qoi.Lower[i],
                                slider_qoi.Upper[i],
                            ),
                            step=slider_step,
                            key="qoi_slider" + str(i),
                            format=f"%0.{int(np.abs(np.log10(slider_step)))}f"
                        )
                    )

            # Since nominal is above the sliders, save step in session state
            if "slider_steps" not in st.session_state:
                st.session_state["slider_steps"] = slider_steps

            submitted = st.form_submit_button("Update")

            if not submitted:
                return st.session_state["updated_dv"], st.session_state["updated_qoi"]

            if submitted:
                st.session_state["updated_dv"] = slider_dv
                st.session_state["updated_qoi"] = slider_qoi
                # Return the updated bounds
                return slider_dv, slider_qoi

    # This way the colors do not change for a given session as the rectangles are manipulated
    def generate_colors(self):
        # Generating unique colors for the quantities of interest
        num_colors = len(self.problem_qoi.Variables)
        # Sample nice colors using seaborns built in color pallette
        cv = np.array(sb.color_palette(n_colors=num_colors))
        cv[:, 1] = 0
        return cv

    def initiate_plots(self):
        # Check if plot matrix is selected
        if not self.plot_matrix:
            self.plotter = self.prob.plotter
        self.plot_rows = int(((self.plotter.shape[0]) - 1) / 3) + 1
        # Generate colors for qoi constraints plotting if not already generated
        if "cv" not in st.session_state:
            cv = self.generate_colors()
            st.session_state["cv"] = cv
        # Create figures with prescribed formatting
        fig = {}
        plot_index = 0
        plotter = self.plotter
        for plot_index in range(len(self.plotter)):
            # Create figures
            fig[plot_index] = go.Figure()
            # Setting the layout of the figures
            if len(self.plotter[plot_index]) == 2:
                fig[plot_index].update_layout(
                    margin=self.plot_margin,
                    autosize=False,
                    width=self.plotsx,
                    height=self.plotsy,
                    xaxis_range=[
                        self.problem_dv.Lower[self.plotter[plot_index][0]],
                        self.problem_dv.Upper[self.plotter[plot_index][0]],
                    ],
                    yaxis_range=[
                        self.problem_dv.Lower[self.plotter[plot_index][1]],
                        self.problem_dv.Upper[self.plotter[plot_index][1]],
                    ],
                    # Add title names from the provided DV sheet
                    xaxis_title=dict(
                        text="{}".format(
                            self.problem_dv.Variables[self.plotter[plot_index][0]]
                        ),
                        font_size=15,
                    ),
                    yaxis_title=dict(
                        text="{}".format(
                            self.problem_dv.Variables[self.plotter[plot_index][1]]
                        ),
                        font_size=15,
                    ),
                    font=self.plot_font,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.75,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=15),
                    ),
                )

            # If one wants to visualise solution spaces in 3D
            if len(self.plotter[plot_index]) == 3:
                fig[plot_index].update_layout(
                    scene=dict(
                        xaxis=dict(
                            range=[
                                self.problem_dv.Lower[self.plotter[plot_index][0]],
                                self.problem_dv.Upper[self.plotter[plot_index][0]],
                            ]
                        ),
                        yaxis=dict(
                            range=[
                                self.problem_dv.Lower[self.plotter[plot_index][1]],
                                self.problem_dv.Upper[self.plotter[plot_index][1]],
                            ]
                        ),
                        zaxis=dict(
                            range=[
                                self.problem_dv.Lower[self.plotter[plot_index][2]],
                                self.problem_dv.Upper[self.plotter[plot_index][2]],
                            ]
                        ),
                    ),
                )
        return fig

    # Draws the initial boxes
    def update_rectangles(self, dv, fig):
        plot_index = 0
        width = 1
        for plot_index in range(len(self.plotter)):
            if len(self.plotter[plot_index]) == 2:
                fig[plot_index].add_shape(
                    type="rect",
                    x0=dv.Lower[self.plotter[plot_index][0]],
                    y0=dv.Lower[self.plotter[plot_index][1]],
                    x1=dv.Upper[self.plotter[plot_index][0]],
                    y1=dv.Upper[self.plotter[plot_index][1]],
                    line=dict(
                        color="Black",
                        width=width,
                    ),
                    fillcolor="rgba(0,0,0,0.05)",
                )

                # Highlight design if collapsed to nominal
                if (
                    self.collapse_space == "Nominal"
                    or dv.Lower[self.plotter[plot_index][1]]
                    == dv.Upper[self.plotter[plot_index][1]]
                    and dv.Lower[self.plotter[plot_index][0]]
                    == dv.Upper[self.plotter[plot_index][0]]
                ):
                    width = 10
                    # Also add vertical and horizontal lines highlighting the design
                    fig[plot_index].add_shape(
                        type="line",
                        x0=self.problem_dv.Lower[self.plotter[plot_index][0]],
                        y0=dv.Lower[self.plotter[plot_index][1]],
                        x1=self.problem_dv.Upper[self.plotter[plot_index][0]],
                        y1=dv.Upper[self.plotter[plot_index][1]],
                        line=dict(color="Black", width=4, dash="dot"),
                    )

                    fig[plot_index].add_shape(
                        type="line",
                        x0=dv.Lower[self.plotter[plot_index][0]],
                        y0=self.problem_dv.Lower[self.plotter[plot_index][1]],
                        x1=dv.Upper[self.plotter[plot_index][0]],
                        y1=self.problem_dv.Upper[self.plotter[plot_index][1]],
                        line=dict(color="Black", width=4, dash="dot"),
                    )

            if len(self.plotter[plot_index]) == 3:
                # Define the verticies of a cube mesh
                xrange = [
                    dv.Lower[self.plotter[plot_index][0]],
                    dv.Upper[self.plotter[plot_index][0]],
                ]
                yrange = [
                    dv.Lower[self.plotter[plot_index][1]],
                    dv.Upper[self.plotter[plot_index][2]],
                ]
                zrange = [
                    dv.Lower[self.plotter[plot_index][2]],
                    dv.Upper[self.plotter[plot_index][2]],
                ]
                xx = [xrange[0]] * 4 + [xrange[1]] * 4
                yy = (
                    [yrange[0]] * 2
                    + [yrange[1]] * 2
                    + [yrange[0]] * 2
                    + [yrange[1]] * 2
                )
                zz = [
                    zrange[0],
                    zrange[1],
                    zrange[0],
                    zrange[1],
                    zrange[0],
                    zrange[1],
                    zrange[0],
                    zrange[1],
                ]

                fig[plot_index].add_trace(
                    go.Mesh3d(x=xx, y=yy, z=zz, showscale=False, opacity=0.4)
                )
        return fig

    def plot_figs(self, fig):
        plot_index = 0
        for plot_row in range(self.plot_rows):
            col = st.columns(self.plot_cols)
            for plot_col in range(self.plot_cols):
                if plot_index >= self.plotter.shape[0]:
                    break

                # Add label description
                if self.label_dropdown:
                    with col[plot_col].expander(""):
                        st.write(
                            """
                            {} : {} \n
                            {} : {}
                        """.format(
                                self.problem_dv.Variables[self.plotter[plot_index][0]],
                                self.problem_dv.Description[
                                    self.plotter[plot_index][0]
                                ],
                                self.problem_dv.Variables[self.plotter[plot_index][1]],
                                self.problem_dv.Description[
                                    self.plotter[plot_index][1]
                                ],
                            )
                        )

                # Plotting the solution spaces
                col[plot_col].plotly_chart(fig[plot_index], use_column_width=True)
                plot_index = plot_index + 1

    ############################################################################
    ######################## Sample variables ##################################
    ############################################################################
    def sample_variables(self, dv):
        dv_samples = pd.DataFrame(
            self.rng.uniform(
                dv.Lower, dv.Upper, (self.sample_size, self.problem_dv_size)
            ),
            columns=dv.Variables,
        )
        return dv_samples

    # Evaluate all the methods in the class and are appended to the list
    def compute_qoi(self, dv_samples):
        # Update prob.var with the evaluated qoi
        self.prob._compute_commons(dv_samples)
        # Print a horizontal line for each call
        for method in self.prob.eval_list:
            func = getattr(self.prob, method)
            func()

    ############################################################################
    ######################## Evaluate qoi ######################################
    ############################################################################
    def evaluate(self, dv_samples, plot_index):
        # Continue with generating the masks
        masks = pd.DataFrame()
        mask_ok = pd.DataFrame(np.ones(self.sample_size), dtype=bool).loc[:, 0]
        for i in range(len(self.prob.eval_list)):
            masks[self.prob.eval_list[i]] = (
                self.prob.var[self.prob.eval_list[i]]
                < float(
                    st.session_state.updated_qoi.Upper.loc[
                        st.session_state.updated_qoi.Variables == self.prob.eval_list[i]
                    ].iloc[0]
                )
            ) & (
                self.prob.var[self.prob.eval_list[i]]
                > float(
                    st.session_state.updated_qoi.Lower.loc[
                        st.session_state.updated_qoi.Variables == self.prob.eval_list[i]
                    ].iloc[0]
                )
            )
            mask_ok = np.logical_and(mask_ok, masks[self.prob.eval_list[i]])
        # print(dv_samples)
        dv_samples_ok = dv_samples[mask_ok]
        return masks, dv_samples_ok

    ############################################################################
    ######################## Plotting ##########################################
    ############################################################################
    def compute_mappings(self):
        session_change = not (
            st.session_state.sessionID == st.session_state.prev_sessionID
        )
        slider_change = not (
            (
                st.session_state.updated_dv.Lower
                == st.session_state.prev_updated_dv.Lower
            ).all()
            and (
                st.session_state.updated_dv.Upper
                == st.session_state.prev_updated_dv.Upper
            ).all()
        )
        sample_size_change = not (st.session_state.prev_sample_size == self.sample_size)
        # Resampling everytime makes sense especially when sample_size is small
        # Sample the entire space
        self.dv_samples_full = self.sample_variables(self.problem_dv)
        # Sample only in the updated space through the sliders
        self.dv_samples_slider = self.sample_variables(st.session_state.updated_dv)
        # Now compute the bottom-up mappings
        self.masks = {}
        self.dv_samples_ok = {}
        self.dv_samples_per_plot = {}
        plotter = self.plotter
        for plot_index in range(len(plotter)):
            print("Plot progress: ", str(plot_index + 1) + "/" + str(len(plotter)))
            # Copy the slider samples to a different variable
            self.dv_samples_plot = self.dv_samples_slider.copy()
            # Modify the samples for a particular evaluation
            self.dv_samples_plot.iloc[:, self.plotter[plot_index][0]] = (
                self.dv_samples_full.iloc[:, self.plotter[plot_index][0]]
            )
            self.dv_samples_plot.iloc[:, self.plotter[plot_index][1]] = (
                self.dv_samples_full.iloc[:, self.plotter[plot_index][1]]
            )
            # Evaluate the functions using these updated samples and obtain the masks
            if (
                (session_change == slider_change)
                or sample_size_change
                or self.rerun_button
            ):
                # If there are changes then re-sample and evaluate
                self.compute_qoi(self.dv_samples_plot)
                # Save computed vars per plotindex and reuse them if QoI doesnt change
                self.prob.var.to_csv(
                    self.problem_path + "/output/plot_data_" + str(plot_index) + ".csv",
                    index=False,
                )
            else:
                # Load the saved data
                self.prob.var = pd.read_csv(
                    self.problem_path + "/output/plot_data_" + str(plot_index) + ".csv"
                )
                # Extract the samples and send them for re=masking
                self.dv_samples_plot = self.prob.var.iloc[:, : -self.problem_qoi_size]

            # Update samples per plot
            self.dv_samples_per_plot[plot_index] = self.dv_samples_plot
            self.masks[plot_index], self.dv_samples_ok[plot_index] = self.evaluate(
                self.dv_samples_plot, plot_index
            )
            st.session_state.prev_updated_dv = st.session_state.updated_dv
        # Print a horizontal line after each evaluation
        term_size = os.get_terminal_size()
        print("-" * term_size.columns)
        return self.masks, self.dv_samples_ok

    def scatter_plots(self, fig):
        # Before plotting run mappings
        self.compute_mappings()
        plot_index = 0
        for plot in range(len(self.plotter)):
            if len(self.plotter[plot_index]) == 2:
                trace_ok = go.Scatter(
                    x=self.dv_samples_ok[plot_index].iloc[
                        :, self.plotter[plot_index][0]
                    ],
                    y=self.dv_samples_ok[plot_index].iloc[
                        :, self.plotter[plot_index][1]
                    ],
                    mode="markers",
                    name="feasible designs",
                    marker=self.scatter_markers_ok,
                )

                fig[plot_index].add_trace(trace_ok)
                # Isolating the designs NOT satisfying the constraints
                i = 0
                for idx in self.masks[plot_index].columns:
                    fig[plot_index].add_trace(
                        go.Scatter(
                            x=self.dv_samples_per_plot[plot_index][
                                ~self.masks[plot_index][idx]
                            ].iloc[:, self.plotter[plot_index][0]],
                            y=self.dv_samples_per_plot[plot_index][
                                ~self.masks[plot_index][idx]
                            ].iloc[:, self.plotter[plot_index][1]],
                            mode="markers",
                            name=idx,
                            marker=dict(
                                size=self.marker_size,
                                color="rgba({},{},{},{})".format(
                                    st.session_state.cv[i, 0],
                                    st.session_state.cv[i, 1],
                                    st.session_state.cv[i, 2],
                                    self.qoi_alpha,
                                ),
                            ),
                        )
                    )
                    i = i + 1

            if len(self.plotter[plot_index]) == 3:
                trace_ok = go.Scatter3d(
                    x=self.dv_samples_ok[plot_index].iloc[
                        :, self.plotter[plot_index][0]
                    ],
                    y=self.dv_samples_ok[plot_index].iloc[
                        :, self.plotter[plot_index][1]
                    ],
                    z=self.dv_samples_ok[plot_index].iloc[
                        :, self.plotter[plot_index][2]
                    ],
                    mode="markers",
                    name="feasible designs",
                    marker=self.scatter_markers_ok,
                )

                fig[plot_index].add_trace(trace_ok)
                # Isolating the designs NOT satisfying the constraints
                i = 0
                for idx in self.masks[plot_index].columns:
                    fig[plot_index].add_trace(
                        go.Scatter3d(
                            x=self.dv_samples_per_plot[plot_index][
                                ~self.masks[plot_index][idx]
                            ].iloc[:, self.plotter[plot_index][0]],
                            y=self.dv_samples_per_plot[plot_index][
                                ~self.masks[plot_index][idx]
                            ].iloc[:, self.plotter[plot_index][1]],
                            z=self.dv_samples_per_plot[plot_index][
                                ~self.masks[plot_index][idx]
                            ].iloc[:, self.plotter[plot_index][2]],
                            mode="markers",
                            marker=dict(
                                size=self.marker_size,
                                color="rgba({},{},{},{})".format(
                                    st.session_state.cv[i, 0],
                                    st.session_state.cv[i, 1],
                                    st.session_state.cv[i, 2],
                                    self.qoi_alpha,
                                ),
                            ),
                        )
                    )

            plot_index = plot_index + 1
        return fig

    def overlay_info(self, fig):
        # read the csv file containing all the info
        try:
            self.overlay_info = pd.read_csv(
                self.problem_path + "/input/plot_overlay_i.csv"
            )
            # Remove the first column containing any names
            self.overlay_info = self.overlay_info.iloc[:, 1:]
            # add the traces of a scatter plot with single data point
            plot_index = 0
            for plot in range(len(self.plotter)):
                if len(self.plotter[plot_index]) == 2:
                    # ToDo: very very inefficient fix
                    # ToDo: Cannot handle homogenous data now
                    plot_value1 = self.overlay_info.iloc[:, self.plotter[plot_index][0]]
                    plot_value2 = self.overlay_info.iloc[:, self.plotter[plot_index][1]]
                    # So that plot_value1 is a defined variable as a pandas object
                    if (
                        self.overlay_info.iloc[:, self.plotter[plot_index][0]]
                        .isnull()
                        .any()
                    ):
                        plot_value1.values[:] = self.problem_dv.Lower[
                            self.plotter[plot_index][0]
                        ]
                    if (
                        self.overlay_info.iloc[:, self.plotter[plot_index][1]]
                        .isnull()
                        .any()
                    ):
                        plot_value2.values[:] = self.problem_dv.Lower[
                            self.plotter[plot_index][1]
                        ]

                    trace_info = go.Scatter(
                        x=plot_value1,
                        y=plot_value2,
                        mode="markers",
                        name="Co-optimised design",
                        marker=dict(symbol="circle", color="Black", size=18),
                    )
                    fig[plot_index].add_trace(trace_info)

                plot_index = plot_index + 1
        except IOError:
            pass

        return fig

    ############################################################################
    ######################## Export ##########################################
    ############################################################################
    def export_options(self):
        output_path = self.problem_path + "/output/"
        # To save the generated data and to create an ADG
        export = st.sidebar.expander("Export")
        # Set output path
        with export:
            if export.button("Save solution space"):
                # Saving the solution spaces as csv files
                st.session_state.updated_dv.to_csv(
                    output_path + "dv_solution_space.csv", index=False
                )
                st.session_state.updated_qoi.to_csv(
                    output_path + "qoi_solution_space.csv", index=False
                )
                # Can save plot colors too
                plot_data = {"sample_size": self.sample_size}
                with open(output_path + "plot_data.json", "w") as f:
                    json.dump(plot_data, f)
                st.balloons()

            if export.button("Save plots"):
                fig = st.session_state.figs
                for i in range(len(fig)):
                    fig[i].update_layout(width=800, height=600)
                    pio.write_image(fig[i], output_path + "fig" + str(i) + ".pdf")

            sample_design = export.button("Sample design")
            # if export.button('Sample design'):
        if sample_design:
            sampled_design = self.rng.uniform(
                st.session_state.updated_dv.Lower,
                st.session_state.updated_dv.Upper,
                (1, self.problem_dv_size),
            )
            st.write(
                pd.DataFrame(
                    sampled_design,
                    columns=self.problem_dv.Variables,
                ).transpose()
            )
            st.markdown("```python\n" + repr(sampled_design[0]) + "\n```")

    def load_assets(self):
        load_assets = st.sidebar.expander("Load assets")
        with load_assets:
            st.write("Upload image to show like ADG or image of the system")
            uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            st.image(uploaded_file)

    # def optimisation_options(self):
    #     run_opt = st.sidebar.expander("Optimisation")
    #     with run_opt:
    #         st.write("Runs stochastic iteration optimisation")
    #         if st.button("Optimise"):
    #             pass
