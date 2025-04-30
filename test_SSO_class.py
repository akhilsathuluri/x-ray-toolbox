# %% Example SSO problem for CrashDesign

import numpy as np

# from problems.CrashDesign.library.CrashDesign import CrashDesign
from problems.DSL4RAS.library.DSL4RAS import DSL4RAS
import pandas as pd
import logging
from src.toolopt.XRayOpt import XRayOpt


# %%
sso = XRayOpt(log_level=logging.DEBUG)
sso.growth_rate = 0.1
sso.sample_size = 50
sso.max_exploration_iterations = 5
sso.max_consolidation_iterations = 4
sso._setup_problem(problem_path="problems/DSL4RAS", problem_class=DSL4RAS())
logging.basicConfig(level=logging.DEBUG)
# fmt: off
sso._set_initial_box(np.array([[8.0e-01, 1.5e+00, 1.0e+03, 9.0e-01, 1.0e+02, 1.6e+02, 9.0e-01, 1.0e+01, 3.9e-01, 2.6e-01, 1.0e+02, 2.0e+01]] * 2).T)
# fmt: on
dv_box, box_measure = sso.run_sso_stochastic_iteration()
dv_sol_space = sso.export_optimisation_result(dv_box)

# %%
print(f"Final box measure: {box_measure}")
print(f"Final design variable box: {dv_box}")

# %%
# seed = 42
# rng = np.random.default_rng(seed)

# filetag = "DSL4RAS"
# dv = pd.read_csv(f"problems/{filetag}/input/dv_space.csv", dtype=str)
# qoi = pd.read_csv(f"problems/{filetag}/input/qoi_space.csv", dtype=str)

# dv.iloc[:, 1:3] = dv.iloc[:, 1:3].astype(np.float64)
# qoi.iloc[:, 1:3] = qoi.iloc[:, 1:3].astype(np.float64)

# dv_size = dv.shape[0]
# qoi_size = qoi.shape[0]

# # %% Evaluate the computed samples

# problem = DSL4RAS()


# # %% Now we have the DVs and the QoIs and an easy way to compute them
# """
# Implement the SSO

# Steps:
# Accept an initial guess for the upper and lower bounds of the DVs
# The implementation has two phases: Exploration and Consolidation

# Phase 1: Exploration
# - Set a value for the growth rate


# """
# #  Implementation of Phase-1: Exploration

# is_done = False
# min_growth_rate = 1e-3
# max_growth_rate = 0.2
# growth_rate = 5e-2
# max_exploration_iterations = 20
# max_consolidation_iterations = 20
# ii_exploration = 0
# use_adaptive_growth_rate = False
# min_purity = 1e-3
# max_purity = 0.999
# slack = 0.0
# apply_leanness = True
# sample_size = 100


# def box_measure_volume(dv_box, fraction_useful=1.0):
#     # TODO: Check the size of the DVs and check the size of the design box
#     if dv_box.shape[1] != 2:
#         raise ValueError(
#             "DV box must have exactly two columns for upper and lower bounds."
#         )
#     volume = np.prod(dv_box[:, 1] - dv_box[:, 0]) * fraction_useful
#     return volume


# def dv_box_grow_fixed(dv_box, dv_l, dv_u, growth_rate):
#     dv_box_grown = dv_box + growth_rate * (dv_u - dv_l).reshape(-1, 1) * np.array(
#         [-1, 1]
#     )
#     dv_box_grown[:, 0] = np.clip(dv_box_grown[:, 0], dv_l, dv_u)
#     dv_box_grown[:, 1] = np.clip(dv_box_grown[:, 1], dv_l, dv_u)
#     return dv_box_grown


# def compute_qoi_violation(qoi_evaluated, qoi_l, qoi_u):
#     # TODO: Assert lb==ub shape
#     qoi_violation = np.zeros_like(qoi_evaluated)
#     normalisation_factor = 1.0
#     qoi_violation_l = (qoi_l - qoi_evaluated) / normalisation_factor
#     qoi_violation_u = (qoi_evaluated - qoi_u) / normalisation_factor
#     qoi_violation = np.maximum(qoi_violation_l, qoi_violation_u)
#     return qoi_violation


# def compute_qoi_violation_score(qoi_violation):
#     max_violation = np.max(qoi_violation, axis=1)
#     score = np.zeros_like(max_violation)
#     feasible_mask = (max_violation < 0).astype(bool)
#     score[feasible_mask] = np.mean(qoi_violation[feasible_mask], axis=1)
#     score[~feasible_mask] = np.linalg.norm(qoi_violation[~feasible_mask], axis=1)
#     return score, feasible_mask


# def get_trimming_order(qoi_score):
#     positive_indices = np.where(qoi_score > 0)[0]
#     sorted_indices = np.argsort(qoi_score[positive_indices])
#     sorted_order = positive_indices[sorted_indices]
#     trimming_order = np.vstack((sorted_order, np.flip(sorted_order)))
#     return trimming_order


# def get_trimmed_box(dv_box, box_measure, dv_samples, feasible_mask, trimming_order):
#     if sum(feasible_mask) == len(feasible_mask):
#         logging.debug("All samples are feasible, no need to trim")
#         return dv_box, box_measure
#     else:
#         # dv_box_trimmed, measure_trimmed = trim_dv_box(
#         #     dv_samples, feasible_mask, trimming_order, dv_box
#         # )
#         dv_box_trimmed, measure_trimmed = trim_dv_box_optimized(
#             dv_samples, feasible_mask, trimming_order, dv_box
#         )
#         # evaluate the number of dv_samples now outside the box for stats
#     return dv_box_trimmed, measure_trimmed


# def dvs_in_box(dv_samples, dv_box, dv_violation_score=False):
#     lower_bounds = dv_box[:, 0]
#     upper_bounds = dv_box[:, 1]
#     dv_in_box_mask = np.all(
#         (dv_samples >= lower_bounds) & (dv_samples <= upper_bounds), axis=1
#     )

#     if dv_violation_score:
#         dv_violation_score = np.max(
#             np.maximum(
#                 (dv_box[:, 0] - dv_samples),
#                 (dv_samples - dv_box[:, 1]),
#             )
#             / (dv_box[:, 1] - dv_box[:, 0]),
#             axis=1,
#         )
#         return dv_in_box_mask, dv_violation_score
#     return dv_in_box_mask, 0


# # +++++++++++++++++++++++++++++++++++
# def find_closest_viable_point(
#     dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
# ):
#     dvs_in_box_mask, _ = dvs_in_box(dv_samples, dv_box)
#     if is_lower:
#         remain_region = dv_samples[:, dimension] >= dv_samples[trim_idx, dimension]
#     else:
#         remain_region = dv_samples[:, dimension] <= dv_samples[trim_idx, dimension]

#     viables = dv_samples[
#         np.logical_and(
#             np.logical_and(remain_region, dvs_in_box_mask),
#             feasible_mask,
#         ),
#         dimension,
#     ]
#     if len(viables) > 0:
#         return np.min(viables) if is_lower else np.max(viables)
#     else:
#         return dv_samples[trim_idx, dimension]


# def calculate_box_quality(dv_samples, dv_box, feasible_mask):
#     dvs_in_box_mask, _ = dvs_in_box(dv_samples, dv_box)
#     dvs_in_box_and_feasible = np.logical_and(dvs_in_box_mask, feasible_mask)
#     fraction_useful = sum(dvs_in_box_and_feasible) / len(dvs_in_box_mask)
#     box_measure = box_measure_volume(dv_box, fraction_useful=fraction_useful)
#     return box_measure


# def adjust_box_boundary(
#     dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
# ):
#     adjusted_box = dv_box.copy()
#     boundary_idx = 0 if is_lower else 1

#     closest_viable = dv_samples[trim_idx, dimension]  # Default
#     if slack < 1:
#         closest_viable = find_closest_viable_point(
#             dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
#         )

#     # Apply slack factor
#     adjusted_box[dimension, boundary_idx] = dv_samples[
#         trim_idx, dimension
#     ] * slack + closest_viable * (1 - slack)

#     # Calculate box quality
#     box_measure = calculate_box_quality(dv_samples, adjusted_box, feasible_mask)

#     return adjusted_box, box_measure


# def trim_dv_box_optimized(dv_samples, feasible_mask, trimming_order, dv_box):
#     dv_box_trimmed = dv_box.copy()
#     box_measure_trimmed = -np.inf
#     dv_box_init = dv_box.copy()
#     for ii in range(trimming_order.shape[0]):
#         dv_box_current = dv_box_init.copy()
#         for jj in range(trimming_order.shape[1]):
#             trim_idx = trimming_order[ii, jj]
#             dv_box_best = dv_box_current.copy()
#             box_measure_best = -np.inf
#             for dimension in range(dv_box.shape[0]):
#                 for is_lower in [True, False]:
#                     adjusted_box, box_measure = adjust_box_boundary(
#                         dv_samples,
#                         feasible_mask,
#                         trim_idx,
#                         dv_box_current,
#                         dimension,
#                         is_lower,
#                     )
#                     if box_measure > box_measure_best:
#                         dv_box_best = adjusted_box.copy()
#                         box_measure_best = box_measure
#             dv_box_current = dv_box_best
#         box_measure = calculate_box_quality(dv_samples, dv_box_current, feasible_mask)
#         if box_measure > box_measure_trimmed:
#             dv_box_trimmed = dv_box_current.copy()
#             box_measure_trimmed = box_measure
#     return dv_box_trimmed, box_measure_trimmed


# # +++++++++++++++++++++++++++++++++++


# # def trim_dv_box(dv_samples, feasible_mask, trimming_order, dv_box):
# #     dv_box_trimmed = dv_box.copy()
# #     box_measure_trimmed = -np.inf
# #     dv_box_init = dv_box.copy()
# #     # dvs_in_box_mask_init, _ = dvs_in_box(dv_samples, dv_box)
# #     for ii in range(trimming_order.shape[0]):
# #         dv_box_current = dv_box_init.copy()
# #         for jj in range(trimming_order.shape[1]):
# #             trim_idx = trimming_order[ii, jj]
# #             dv_box_best = []
# #             box_measure_best = -np.inf
# #             for kk in range(dv_box.shape[0]):
# #                 for ll in [0, 1]:
# #                     dv_box = dv_box_current.copy()
# #                     closest_viable = []
# #                     if slack < 1:
# #                         dvs_in_box_mask, _ = dvs_in_box(dv_samples, dv_box)
# #                         if ll == 0:
# #                             remain_region = (
# #                                 dv_samples[:, kk] >= dv_samples[trim_idx, kk]
# #                             )
# #                             viables = dv_samples[
# #                                 np.logical_and(
# #                                     np.logical_and(remain_region, dvs_in_box_mask),
# #                                     feasible_mask,
# #                                 ),
# #                                 kk,
# #                             ]
# #                             if len(viables) > 0:
# #                                 closest_viable = np.min(viables)
# #                             else:
# #                                 closest_viable = dv_samples[trim_idx, kk]
# #                         else:
# #                             remain_region = (
# #                                 dv_samples[:, kk] <= dv_samples[trim_idx, kk]
# #                             )
# #                             viables = dv_samples[
# #                                 np.logical_and(
# #                                     np.logical_and(remain_region, dvs_in_box_mask),
# #                                     feasible_mask,
# #                                 ),
# #                                 kk,
# #                             ]
# #                             if len(viables) > 0:
# #                                 closest_viable = np.max(viables)
# #                             else:
# #                                 closest_viable = dv_samples[trim_idx, kk]

# #                     dv_box[kk, ll] = dv_samples[
# #                         trim_idx, kk
# #                     ] * slack + closest_viable * (1 - slack)
# #                     dvs_in_box_mask, _ = dvs_in_box(dv_samples, dv_box)
# #                     dvs_in_box_and_feasible = np.logical_and(
# #                         dvs_in_box_mask, feasible_mask
# #                     )
# #                     fraction_useful = sum(dvs_in_box_and_feasible) / len(
# #                         dvs_in_box_mask
# #                     )
# #                     box_measure = box_measure_volume(
# #                         dv_box, fraction_useful=fraction_useful
# #                     )
# #                     if box_measure > box_measure_best:
# #                         dv_box_best = dv_box.copy()
# #                         box_measure_best = box_measure.copy()
# #                     # breakpoint()
# #             dv_box_current = dv_box_best

# #         dvs_in_box_mask, _ = dvs_in_box(dv_samples, dv_box_current)
# #         dvs_in_box_and_feasible = np.logical_and(dvs_in_box_mask, feasible_mask)
# #         fraction_useful = sum(dvs_in_box_and_feasible) / len(dvs_in_box_mask)
# #         box_measure = box_measure_volume(
# #             dv_box_current, fraction_useful=fraction_useful
# #         )
# #         if box_measure > box_measure_trimmed:
# #             dv_box_trimmed = dv_box_current
# #             box_measure_trimmed = box_measure

# #     return dv_box_trimmed, box_measure_trimmed


# # Initial guess for the design box
# dv_box_init = np.column_stack(
#     [((dv.Upper + dv.Lower) / 2).to_numpy().astype(np.float64)] * 2
# )
# # dv_box_init = np.column_stack([[4.05e5, 2e3, 4.05e5, 15.6, 0.3, 0.3]] * 2)
# box_measure_init = box_measure_volume(dv_box_init)
# # dv_box_grown = dv_box_grow_fixed(
# #     dv_box_init,
# #     dv.Lower.to_numpy().astype(np.float64),
# #     dv.Upper.to_numpy().astype(np.float64),
# #     growth_rate=growth_rate,
# # )
# # dv_samples = pd.DataFrame(
# #     rng.uniform(dv_box_grown[:, 0], dv_box_grown[:, 1], size=(sample_size, dv_size)),
# #     columns=dv.Variables,
# # )
# # qoi_violation = compute_qoi_violation(
# #     problem.var[qoi.Variables].to_numpy().astype(np.float64),
# #     qoi.Lower.to_numpy().astype(np.float64),
# #     qoi.Upper.to_numpy().astype(np.float64),
# # )
# # qoi_score, feasible_mask = compute_qoi_violation_score(qoi_violation)
# # trimming_order = get_trimming_order(qoi_score)
# # dv_box_trimmed, box_measure_trimmed = get_trimmed_box(
# #     dv_box_grown,
# #     box_measure,
# #     dv_samples[dv.Variables].to_numpy().astype(np.float64),
# #     feasible_mask,
# #     trimming_order,
# # )

# # print("dv_box_trimmed", dv_box_trimmed)
# # print("box_measure_trimmed", box_measure_trimmed)
# logging.debug(f"Initial box measure: {box_measure_init}")

# # %%
# # box_measure_trimmed = box_measure_init
# # dv_box = dv_box_init
# # for ii_exploration in range(max_exploration_iterations):
# #     # Modification step B - Extend the candidate box
# #     if ii_exploration > 1 and use_adaptive_growth_rate:
# #         # logging.debug("Adaptive growth rate")
# #         # purity = np.clip(purity, min_purity, max_purity)
# #         # delta_box_measure = box_measure - prev_box_measure
# #         raise NotImplementedError("Adaptive growth rate is not implemented yet")
# #         break
# #     prev_box_measure = box_measure_trimmed
# #     logging.debug(f"Iteration {ii_exploration} - Box measure: {box_measure_trimmed}")
# #     logging.debug("Growing candidate box")

# #     # grow the candidate box
# #     dv_box_grown = dv_box_grow_fixed(
# #         dv_box,
# #         dv.Lower.to_numpy().astype(np.float64),
# #         dv.Upper.to_numpy().astype(np.float64),
# #         growth_rate=growth_rate,
# #     )
# #     logging.debug(f"DV box grown: {dv_box_grown}")
# #     logging.debug(f"Current growth rate is: {growth_rate}")

# #     # generate samples within the design box
# #     dv_samples = pd.DataFrame(
# #         rng.uniform(
# #             dv_box_grown[:, 0], dv_box_grown[:, 1], size=(sample_size, dv_size)
# #         ),
# #         columns=dv.Variables,
# #     )

# #     # evaluate the samples for all the QoIs
# #     problem._compute_commons(dv_samples)
# #     for method in qoi.Variables:
# #         func = getattr(problem, method)
# #         func()

# #     # generate labels and scores
# #     qoi_violation = compute_qoi_violation(
# #         problem.var[qoi.Variables].to_numpy().astype(np.float64),
# #         qoi.Lower.to_numpy().astype(np.float64),
# #         qoi.Upper.to_numpy().astype(np.float64),
# #     )
# #     qoi_score, feasible_mask = compute_qoi_violation_score(qoi_violation)
# #     if sum(feasible_mask) == 0:
# #         logging.debug("No feasible samples found, relax constraints and retry")

# #     purity = sum(feasible_mask) / sample_size
# #     box_measure_grown = box_measure_volume(
# #         dv_box_grown, fraction_useful=sum(feasible_mask) / sample_size
# #     )
# #     logging.debug(f"Purity: {purity}")

# #     if sum(feasible_mask) == 0 or purity < min_purity:
# #         logging.debug("Purity is below the threshold, reducing growth rate")

# #     # Modification step A - Remove bad sample designs
# #     # get trimming order
# #     trimming_order = get_trimming_order(qoi_score)
# #     dv_box_trimmed, box_measure_trimmed = get_trimmed_box(
# #         dv_box_grown,
# #         box_measure_grown,
# #         dv_samples[dv.Variables].to_numpy().astype(np.float64),
# #         feasible_mask,
# #         trimming_order,
# #     )
# #     logging.debug(f"Box measure after trimming: {box_measure_trimmed}")
# #     if apply_leanness:
# #         # TODO: Need to add this
# #         pass

# #     dv_box = dv_box_trimmed

# #     logging.debug(f"DV box trimmed: {dv_box_trimmed}")
# #     logging.debug(f"Current box measure: {box_measure_trimmed}")
# # logging.debug(f"Exploration phase completed. Box measure: {box_measure_trimmed}")


# # for ii_consolidation in range(max_consolidation_iterations):
# #     logging.debug(f"Consolidation iteration {ii_consolidation}")
# #     dv_samples = pd.DataFrame(
# #         rng.uniform(dv_box[:, 0], dv_box[:, 1], size=(sample_size, dv_size)),
# #         columns=dv.Variables,
# #     )
# #     problem._compute_commons(dv_samples)
# #     for method in qoi.Variables:
# #         func = getattr(problem, method)
# #         func()
# #     qoi_violation = compute_qoi_violation(
# #         problem.var[qoi.Variables].to_numpy().astype(np.float64),
# #         qoi.Lower.to_numpy().astype(np.float64),
# #         qoi.Upper.to_numpy().astype(np.float64),
# #     )
# #     qoi_score, feasible_mask = compute_qoi_violation_score(qoi_violation)
# #     if sum(feasible_mask) == 0:
# #         logging.debug("No feasible samples found, relax constraints and retry")

# #     box_measure = box_measure_volume(
# #         dv_box, fraction_useful=sum(feasible_mask) / sample_size
# #     )

# #     # Modification step A - Remove bad sample designs
# #     trimming_order = get_trimming_order(qoi_score)
# #     dv_box_trimmed, box_measure_trimmed = get_trimmed_box(
# #         dv_box,
# #         box_measure,
# #         dv_samples[dv.Variables].to_numpy().astype(np.float64),
# #         feasible_mask,
# #         trimming_order,
# #     )
# #     logging.debug(f"Box measure after trimming: {box_measure_trimmed}")
# #     if apply_leanness:
# #         # TODO: Need to add this
# #         pass

# # dv_box = dv_box_trimmed
# # logging.debug(f"Consolidation phase completed. Box measure: {box_measure_trimmed}")
# # logging.debug(f"Final DV box: {dv_box}")


# def evaluate_and_trim_box(dv_box, sample_size, is_exploration=False, growth_rate=None):
#     """Common functionality to evaluate samples and trim the design box"""
#     # If exploration phase, grow the box first
#     working_box = dv_box
#     if is_exploration:
#         logging.debug("Growing candidate box")
#         working_box = dv_box_grow_fixed(
#             dv_box,
#             dv.Lower.to_numpy().astype(np.float64),
#             dv.Upper.to_numpy().astype(np.float64),
#             growth_rate=growth_rate,
#         )
#         logging.debug(f"DV box grown: {working_box}")
#         logging.debug(f"Current growth rate is: {growth_rate}")

#     # Generate samples within the design box
#     dv_samples = pd.DataFrame(
#         rng.uniform(working_box[:, 0], working_box[:, 1], size=(sample_size, dv_size)),
#         columns=dv.Variables,
#     )

#     # Evaluate the samples for all the QoIs
#     problem._compute_commons(dv_samples)
#     for method in qoi.Variables:
#         func = getattr(problem, method)
#         func()

#     # Generate labels and scores
#     qoi_violation = compute_qoi_violation(
#         problem.var[qoi.Variables].to_numpy().astype(np.float64),
#         qoi.Lower.to_numpy().astype(np.float64),
#         qoi.Upper.to_numpy().astype(np.float64),
#     )
#     qoi_score, feasible_mask = compute_qoi_violation_score(qoi_violation)
#     if sum(feasible_mask) == 0:
#         logging.debug("No feasible samples found, relax constraints and retry")

#     # Calculate purity and box measure
#     purity = sum(feasible_mask) / sample_size
#     box_measure = box_measure_volume(working_box, fraction_useful=purity)

#     if is_exploration:
#         logging.debug(f"Purity: {purity}")
#         if sum(feasible_mask) == 0 or purity < min_purity:
#             logging.debug("Purity is below the threshold, reducing growth rate")

#     # Trim the box
#     trimming_order = get_trimming_order(qoi_score)
#     dv_box_trimmed, box_measure_trimmed = get_trimmed_box(
#         working_box,
#         box_measure,
#         dv_samples[dv.Variables].to_numpy().astype(np.float64),
#         feasible_mask,
#         trimming_order,
#     )
#     logging.debug(f"Box measure after trimming: {box_measure_trimmed}")

#     # Apply leanness if requested
#     if apply_leanness:
#         # TODO: Need to add this
#         pass

#     return dv_box_trimmed, box_measure_trimmed


# # Single loop for both exploration and consolidation phases
# box_measure_trimmed = box_measure_init
# dv_box = dv_box_init
# total_iterations = max_exploration_iterations + max_consolidation_iterations

# for iteration in range(total_iterations):
#     # Determine which phase we're in
#     is_exploration_phase = iteration < max_exploration_iterations
#     phase_name = "Exploration" if is_exploration_phase else "Consolidation"
#     phase_iteration = (
#         iteration if is_exploration_phase else iteration - max_exploration_iterations
#     )

#     logging.debug(f"{phase_name} iteration {phase_iteration}")

#     # Handle adaptive growth rate in exploration phase
#     if is_exploration_phase and iteration > 1 and use_adaptive_growth_rate:
#         # logging.debug("Adaptive growth rate")
#         # purity = np.clip(purity, min_purity, max_purity)
#         # delta_box_measure = box_measure - prev_box_measure
#         raise NotImplementedError("Adaptive growth rate is not implemented yet")
#         break

#     prev_box_measure = box_measure_trimmed

#     if is_exploration_phase:
#         logging.debug(
#             f"Iteration {phase_iteration} - Box measure: {box_measure_trimmed}"
#         )

#     # Evaluate and trim the box
#     dv_box, box_measure_trimmed = evaluate_and_trim_box(
#         dv_box,
#         sample_size,
#         is_exploration=is_exploration_phase,
#         growth_rate=growth_rate if is_exploration_phase else None,
#     )

#     # Log phase transition
#     if is_exploration_phase and iteration == max_exploration_iterations - 1:
#         logging.debug(
#             f"Exploration phase completed. Box measure: {box_measure_trimmed}"
#         )

#     if not is_exploration_phase and iteration == total_iterations - 1:
#         logging.debug(
#             f"Consolidation phase completed. Box measure: {box_measure_trimmed}"
#         )
#         logging.debug(f"Final DV box: {dv_box}")


# # %%
# # Export the final DV box to a CSV file
# dv_solution_space = pd.DataFrame(
#     {
#         "Variables": dv["Variables"],
#         "Lower": dv_box[:, 0],
#         "Upper": dv_box[:, 1],
#         "Units": dv["Units"],
#         "Description": dv["Description"],
#     }
# )
# dv_solution_space.to_csv(
#     "problems/CrashDesign/output/dv_solution_space.csv", index=False
# )
# logging.debug(
#     "Exported DV solution space to problems/CrashDesign/output/dv_solution_space.csv"
# )
