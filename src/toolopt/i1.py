#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSO_BOX_STOCHASTIC Box-shaped solution spaces optimization (Stochastic method)

SSO_BOX_STOCHASTIC uses a modified version of the stochastic method to
compute optimal solution (or requirement) spaces.

This Python implementation corresponds to the MATLAB version with the same functionality.

Usage:
    candidate_box, problem_data, iteration_data = sso_box_stochastic(
        design_evaluator, initial_box, design_space_lower_bound,
        design_space_upper_bound, **options)

Input:
    - design_evaluator : DesignEvaluatorBase
    - initial_box : numpy.ndarray (1,n_design_variable) OR (2,n_design_variable)
    - design_space_lower_bound : numpy.ndarray (1,n_design_variable)
    - design_space_upper_bound : numpy.ndarray (1,n_design_variable)
    - options : dictionary

Output:
    - candidate_box : numpy.ndarray (2,n_design_variable)
    - problem_data : dict
        - design_evaluator : DesignEvaluatorBase
        - initial_box : numpy.ndarray (2,n_design_variable)
        - design_space_lower_bound : numpy.ndarray (1,n_design_variable)
        - design_space_upper_bound : numpy.ndarray (1,n_design_variable)
        - options : dict
        - initial_rng_state : numpy.random.Generator
    - iteration_data : list of dict
        - evaluated_design_samples : numpy.ndarray (n_sample,n_design_variable)
        - evaluation_output : class-dependent
        - phase : int
        - growth_rate : float
        - design_score : numpy.ndarray (n_sample,1)
        - is_good_performance : numpy.ndarray (n_sample,1) boolean
        - is_physically_feasible : numpy.ndarray (n_sample,1) boolean
        - is_acceptable : numpy.ndarray (n_sample,1) boolean
        - is_useful : numpy.ndarray (n_sample,1) boolean
        - candidate_box_before_trim : numpy.ndarray (2,n_design_variable)
        - candidate_box_after_trim : numpy.ndarray (2,n_design_variable)

See also: sso_stochastic_options.

Copyright 2025 Eduardo Rodrigues Della Noce
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import time
import random
from typing import Tuple, Dict, List, Any, Optional, Callable, Union


def sso_box_stochastic(
    design_evaluator,
    initial_box,
    design_space_lower_bound,
    design_space_upper_bound,
    **kwargs,
) -> Tuple[np.ndarray, Optional[Dict], Optional[List[Dict]]]:
    """
    Box-shaped solution spaces optimization using stochastic method.

    Computes optimal solution (or requirement) spaces within defined design space.

    Args:
        design_evaluator: A design evaluator object that can evaluate design points
        initial_box: Initial box for the algorithm, either 1 or 2 rows
        design_space_lower_bound: Lower bound of the design space
        design_space_upper_bound: Upper bound of the design space
        **kwargs: Additional options for the algorithm

    Returns:
        candidate_box: Optimized solution space box
        problem_data: Data about the problem setup (if requested)
        iteration_data: Data from each iteration (if requested)
    """
    # Options
    default_options = sso_stochastic_options("box")
    options = {**default_options, **kwargs}

    # Extract options as necessary
    # Requirement spaces
    requirement_spaces_type = options.get("requirement_spaces_type")
    apply_leanness_each_trim = options.get("apply_leanness", "").lower() == "always"
    apply_leanness_final_trim = options.get("apply_leanness", "").lower() in [
        "always",
        "end-only",
    ]

    # Trimming
    trimming_operation_options = {
        "measure_function": options.get("measure_function"),
        "measure_options": options.get("measure_options", {}),
        **options.get("trimming_operation_options", {}),
    }

    # Logging verbosity
    console = ConsoleLogging(options.get("logging_level", "info"))

    # Initial Candidate Box
    if initial_box.shape[0] == 1:
        candidate_box = np.vstack([initial_box, initial_box])  # single point
    elif initial_box.shape[0] == 2:
        candidate_box = initial_box  # candidate box
    else:
        console.error(
            "SSOBoxOptStochastic:InitialGuessWrong",
            "Error. Initial guess incompatible in sso_box_stochastic.",
        )
        raise ValueError("Initial box must have 1 or 2 rows")

    # Initial Measure
    measure_trimmed = options["measure_function"](
        candidate_box, None, **options.get("measure_options", {})
    )
    if np.isinf(measure_trimmed) or np.isnan(measure_trimmed):
        measure_trimmed = 0
    n_dimension = design_space_lower_bound.shape[1]

    # Log Initialization
    output_problem_data = kwargs.get("output_problem_data", True)
    if output_problem_data:
        problem_data = {
            "design_evaluator": design_evaluator,
            "initial_box": candidate_box,
            "design_space_lower_bound": design_space_lower_bound,
            "design_space_upper_bound": design_space_upper_bound,
            "options": options,
            "initial_rng_state": np.random.get_state(),
        }
    else:
        problem_data = None

    output_iteration_data = kwargs.get("output_iteration_data", True)
    if output_iteration_data:
        iteration_data = []
    else:
        iteration_data = None

    # Phase I - Exploration
    i_exploration = 1
    growth_rate = options.get("growth_rate", 0.1)
    has_converged_exploration = False

    while (not has_converged_exploration) and (
        i_exploration <= options.get("max_iter_exploration", 10)
    ):
        # Modification Step B - Growth: Extend Candidate Box
        console.info("=" * 120)
        console.info(f"Initiating Phase I - Exploration: Iteration {i_exploration}")

        # Change growth rate depending on previous result
        if i_exploration > 1 and options.get("use_adaptive_growth_rate", False):
            console.info("Adapting growth rate... ")
            start_time = time.time()

            purity = n_acceptable / n_sample
            purity = max(
                min(purity, options.get("maximum_growth_purity", 1.0)),
                options.get("minimum_growth_purity", 0.0),
            )

            increase_measure = measure_grown - measure_previous
            increase_measure_acceptable = max(
                measure_grown * purity - measure_previous, 0
            )
            fraction_acceptable_increase_measure = (
                increase_measure_acceptable / increase_measure
                if increase_measure != 0
                else 0
            )

            # Change step size based on purity compared to target
            growth_adaptation_factor = options["growth_adaptation_factor_function"](
                purity,
                options.get("target_accepted_ratio_exploration", 0.5),
                n_dimension,
                fraction_acceptable_increase_measure,
                **options.get("growth_adaptation_factor_options", {}),
            )

            growth_adaptation_factor = max(
                min(
                    growth_adaptation_factor,
                    options.get("maximum_growth_adaptation_factor", 2.0),
                ),
                options.get("minimum_growth_adaptation_factor", 0.5),
            )

            growth_rate = growth_adaptation_factor * growth_rate
            growth_rate = max(
                min(growth_rate, options.get("maximum_growth_rate", 0.5)),
                options.get("minimum_growth_rate", 0.01),
            )

            console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

        measure_previous = measure_trimmed

        # Where design variables aren't fixed, expand candidate solution box
        # in both sides of each interval isotropically
        console.info("Growing candidate box... ")
        start_time = time.time()

        candidate_box_grown = design_box_grow_fixed(
            candidate_box,
            design_space_lower_bound,
            design_space_upper_bound,
            growth_rate,
        )

        console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")
        console.debug(f"- Current Growth Rate: {growth_rate}")

        # Sample inside the current candidate box
        # Get current number of samples
        n_sample = get_current_array_entry(
            options.get("number_samples_per_iteration_exploration", [100]),
            i_exploration,
        )

        # Generate samples that are to be evaluated
        design_sample = sso_box_sub_generate_new_sample_points(
            candidate_box_grown,
            n_sample,
            options.get("sampling_method_function"),
            options.get("sampling_method_options", {}),
            console,
        )

        # Evaluate the samples
        is_good_performance, is_physically_feasible, score, output_evaluation = (
            sso_box_sub_evaluate_sample_points(design_evaluator, design_sample, console)
        )

        # Label samples according to desired requirement spaces problem type
        is_acceptable, is_useful = sso_box_sub_label_samples_requirement_spaces(
            requirement_spaces_type,
            is_good_performance,
            is_physically_feasible,
            console,
        )

        # Count number of labels
        n_acceptable, n_useful, n_acceptable_useful = (
            sso_box_sub_count_label_acceptable_useful(is_acceptable, is_useful, console)
        )

        # Compute candidate box measure
        measure_grown = sso_box_sub_compute_candidate_box_measure(
            candidate_box_grown,
            n_sample,
            n_useful,
            options.get("measure_function"),
            options.get("measure_options", {}),
            console,
        )

        # Box may have grown too much + "unlucky sampling" w/ no good
        # points, go back in this case
        if n_acceptable == 0 or n_useful == 0 or n_acceptable_useful == 0:
            console.warn(
                "SSOOptBox:BadSampling",
                "No good/useful points found, rolling back and reducing growth rate to minimum...",
            )

            if output_iteration_data:
                console.info("Logging relevant information... ")
                start_time = time.time()

                iteration_data.append(
                    {
                        # System data
                        "evaluated_design_samples": design_sample,
                        "evaluation_output": output_evaluation,
                        # Algorithm data
                        "phase": 1,
                        "growth_rate": growth_rate,
                        # Problem data
                        "design_score": score,
                        "is_good_performance": is_good_performance,
                        "is_physically_feasible": is_physically_feasible,
                        "is_acceptable": is_acceptable,
                        "is_useful": is_useful,
                        # Trimming data
                        "candidate_box_before_trim": candidate_box_grown,
                        "candidate_box_after_trim": candidate_box,
                    }
                )

                console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

            # Prepare for next iteration
            i_exploration += 1
            continue

        # Modification Step A - Trimming: Remove Bad Points
        # Find trimming order
        order_trim = options["trimming_order_function"](
            ~is_acceptable, score, **options.get("trimming_order_options", {})
        )

        candidate_box_trimmed, measure_trimmed = sso_box_sub_trimming_operation(
            candidate_box_grown,
            measure_grown,
            design_sample,
            is_acceptable,
            is_useful,
            order_trim,
            options["trimming_operation_function"],
            trimming_operation_options,
            console,
        )

        if apply_leanness_each_trim:
            trimming_order = trimming_order(
                ~is_useful, score, order_preference="score-low-to-high"
            )
            candidate_box_trimmed = box_trimming_leanness(
                design_sample, is_useful, trimming_order, candidate_box_trimmed
            )
            # Get estimate of new measure
            inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
            measure_trimmed = sso_box_sub_compute_candidate_box_measure(
                candidate_box,
                np.sum(inside_box_new),
                np.sum(inside_box_new & is_useful),
                options.get("measure_function"),
                options.get("measure_options", {}),
                console,
            )

        # Convergence Criteria
        console.info("Checking convergence... ")
        start_time = time.time()

        # Stop phase I if measure doesn't change significantly from step to step
        if i_exploration >= options.get("max_iter_exploration", 10):
            has_converged_exploration = True
        elif not options.get("fix_iter_number_exploration", False) and abs(
            measure_trimmed - measure_previous
        ) / measure_trimmed < options.get("tolerance_measure_change_exploration", 0.01):
            has_converged_exploration = True

        console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

        if output_iteration_data:
            # Save Data
            console.info("Logging relevant information... ")
            start_time = time.time()

            iteration_data.append(
                {
                    # System data
                    "evaluated_design_samples": design_sample,
                    "evaluation_output": output_evaluation,
                    # Algorithm data
                    "phase": 1,
                    "growth_rate": growth_rate,
                    # Problem data
                    "design_score": score,
                    "is_good_performance": is_good_performance,
                    "is_physically_feasible": is_physically_feasible,
                    "is_acceptable": is_acceptable,
                    "is_useful": is_useful,
                    # Trimming data
                    "candidate_box_before_trim": candidate_box_grown,
                    "candidate_box_after_trim": candidate_box_trimmed,
                }
            )

            console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

        # Prepare for next iteration
        console.info(f"Done with iteration {i_exploration}!\n")

        candidate_box = candidate_box_trimmed
        i_exploration += 1

    console.info(f"\nDone with Phase I - Exploration in iteration {i_exploration-1}!\n")

    # Intermediary: Sample inside current (End of Exploration) candidate box
    console.info("=" * 120)
    console.info("Initiating transition to Phase II - Consolidation...")

    # Phase II - Consolidation
    # Iteration start
    if (
        options.get("fix_iter_number_consolidation", False)
        and options.get("max_iter_consolidation", 5) == 0
    ):
        convergence_consolidation = True
    else:
        convergence_consolidation = False

    i_consolidation = 1
    while (not convergence_consolidation) and (
        i_consolidation <= options.get("max_iter_consolidation", 5)
    ):
        console.info("=" * 120)
        console.info(
            f"Initiating Phase II - Consolidation: Iteration {i_consolidation}..."
        )

        # Get current number of samples
        n_sample = get_current_array_entry(
            options.get("number_samples_per_iteration_consolidation", [100]),
            i_consolidation,
        )

        # Generate samples that are to be evaluated
        design_sample = sso_box_sub_generate_new_sample_points(
            candidate_box,
            n_sample,
            options.get("sampling_method_function"),
            options.get("sampling_method_options", {}),
            console,
        )

        # Evaluate the samples
        is_good_performance, is_physically_feasible, score, output_evaluation = (
            sso_box_sub_evaluate_sample_points(design_evaluator, design_sample, console)
        )

        # Label samples according to desired requirement spaces problem type
        is_acceptable, is_useful = sso_box_sub_label_samples_requirement_spaces(
            options.get("requirement_spaces_type"),
            is_good_performance,
            is_physically_feasible,
            console,
        )

        # Count number of labels
        n_acceptable, n_useful, n_acceptable_useful = (
            sso_box_sub_count_label_acceptable_useful(is_acceptable, is_useful, console)
        )

        # No viable design found; throw error
        if n_acceptable == 0 or n_useful == 0 or n_acceptable_useful == 0:
            error_msg = (
                "No good/useful points found, please retry process with "
                "different parameters / looser requirements."
            )
            console.critical("SSOOptBox:BadSampling", error_msg)
            raise RuntimeError(error_msg)

        # Compute candidate box measure
        measure = sso_box_sub_compute_candidate_box_measure(
            candidate_box,
            n_sample,
            n_useful,
            options.get("measure_function"),
            options.get("measure_options", {}),
            console,
        )

        # Convergence Check - Purity
        tolerance_purity_consolidation = options.get(
            "tolerance_purity_consolidation", 0.95
        )
        if (
            not options.get("fix_iter_number_consolidation", False)
            and n_acceptable / n_sample >= tolerance_purity_consolidation
        ):
            convergence_consolidation = True

        # Modification Step A (Trimming): Remove Bad Points
        if not convergence_consolidation:
            order_trim = options["trimming_order_function"](
                ~is_acceptable, score, **options.get("trimming_order_options", {})
            )

            candidate_box_trimmed, measure_trimmed = sso_box_sub_trimming_operation(
                candidate_box,
                measure,
                design_sample,
                is_acceptable,
                is_useful,
                order_trim,
                options["trimming_operation_function"],
                options.get("trimming_operation_options", {}),
                console,
            )
        else:
            candidate_box_trimmed = candidate_box
            measure_trimmed = measure

        if apply_leanness_each_trim:
            trimming_order = trimming_order(
                ~is_useful, score, order_preference="score-low-to-high"
            )
            candidate_box_trimmed = box_trimming_leanness(
                design_sample, is_useful, trimming_order, candidate_box_trimmed
            )
            # Get estimate of new measure
            inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
            measure_trimmed = sso_box_sub_compute_candidate_box_measure(
                candidate_box,
                np.sum(inside_box_new),
                np.sum(inside_box_new & is_useful),
                options.get("measure_function"),
                options.get("measure_options", {}),
                console,
            )

        # Convergence check - Number of Iterations
        if i_consolidation >= options.get("max_iter_consolidation", 5):
            convergence_consolidation = True

        if output_iteration_data:
            # Save Data
            console.info("Logging relevant information... ")
            start_time = time.time()

            iteration_data.append(
                {
                    # System data
                    "evaluated_design_samples": design_sample,
                    "evaluation_output": output_evaluation,
                    # Algorithm data
                    "phase": 2,
                    "growth_rate": None,
                    # Problem data
                    "design_score": score,
                    "is_good_performance": is_good_performance,
                    "is_physically_feasible": is_physically_feasible,
                    "is_acceptable": is_acceptable,
                    "is_useful": is_useful,
                    # Trimming data
                    "candidate_box_before_trim": candidate_box,
                    "candidate_box_after_trim": candidate_box_trimmed,
                }
            )

            console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

        # Prepare for next iteration
        console.info(f"Done with iteration {i_consolidation}!\n")
        candidate_box = candidate_box_trimmed
        i_consolidation += 1

    console.info(
        f"\nDone with Phase II - Consolidation in iteration {i_consolidation-1}!\n"
    )

    # Check for the leanness condition
    if apply_leanness_final_trim:
        trimming_order = trimming_order(
            ~is_useful, score, order_preference="score-low-to-high"
        )
        candidate_box = box_trimming_leanness(
            design_sample, is_useful, trimming_order, candidate_box
        )

    return candidate_box, problem_data, iteration_data


# Helper functions for the main algorithm
def sso_box_sub_generate_new_sample_points(
    candidate_box, n_sample, sampling_function, sampling_options, console
):
    """Generate new sample points in the candidate box."""
    console.info("Generating new sample points in candidate box... ")
    start_time = time.time()

    design_sample = sampling_function(candidate_box, n_sample, **sampling_options)

    console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")
    console.debug(f"- Number of samples generated: {n_sample}")

    return design_sample


def sso_box_sub_evaluate_sample_points(design_evaluator, design_sample, console):
    """Evaluate design sample points using the provided evaluator."""
    console.info("Evaluating sample points... ")
    start_time = time.time()

    performance_deficit, physical_feasibility_deficit, output_evaluation = (
        design_evaluator.evaluate(design_sample)
    )

    is_good_performance, score = design_deficit_to_label_score(performance_deficit)

    if (
        physical_feasibility_deficit is not None
        and len(physical_feasibility_deficit) > 0
    ):
        is_physically_feasible, _ = design_deficit_to_label_score(
            physical_feasibility_deficit
        )
    else:
        is_physically_feasible = np.ones(design_sample.shape[0], dtype=bool)

    console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")
    n_samples = design_sample.shape[0]
    good_count = np.sum(is_good_performance)
    console.debug(
        f"- Number of good samples: {good_count} ({100*good_count/n_samples:.1f}%)"
    )
    console.debug(
        f"- Number of bad samples: {n_samples-good_count} ({100*(n_samples-good_count)/n_samples:.1f}%)"
    )

    feasible_count = np.sum(is_physically_feasible)
    console.debug(
        f"- Number of physically feasible samples: {feasible_count} ({100*feasible_count/n_samples:.1f}%)"
    )
    console.debug(
        f"- Number of physically infeasible samples: {n_samples-feasible_count} ({100*(n_samples-feasible_count)/n_samples:.1f}%)"
    )

    return is_good_performance, is_physically_feasible, score, output_evaluation


def sso_box_sub_label_samples_requirement_spaces(
    requirement_spaces_type, is_good_performance, is_physically_feasible, console
):
    """Create labels for design samples based on requirement spaces type."""
    console.info("Creating labels for each design... ")
    start_time = time.time()

    is_acceptable, is_useful = design_requirement_spaces_label(
        requirement_spaces_type, is_good_performance, is_physically_feasible
    )

    console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")
    n_sample = len(is_good_performance)

    acceptable_count = np.sum(is_acceptable)
    console.debug(
        f"- Number of accepted samples: {acceptable_count} ({100*acceptable_count/n_sample:.1f}%)"
    )
    console.debug(
        f"- Number of rejected samples: {n_sample-acceptable_count} ({100*(n_sample-acceptable_count)/n_sample:.1f}%)"
    )

    useful_count = np.sum(is_useful)
    console.debug(
        f"- Number of useful samples: {useful_count} ({100*useful_count/n_sample:.1f}%)"
    )
    console.debug(
        f"- Number of useless samples: {n_sample-useful_count} ({100*(n_sample-useful_count)/n_sample:.1f}%)"
    )

    return is_acceptable, is_useful


def sso_box_sub_count_label_acceptable_useful(is_acceptable, is_useful, console):
    """Count the number of acceptable and useful samples."""
    n_acceptable = np.sum(is_acceptable)
    n_useful = np.sum(is_useful)
    n_acceptable_useful = np.sum(is_acceptable & is_useful)

    n_samples = len(is_acceptable)
    console.debug(
        f"- Number of accepted samples: {n_acceptable} ({100*n_acceptable/n_samples:.1f}%)"
    )
    console.debug(
        f"- Number of useful samples: {n_useful} ({100*n_useful/n_samples:.1f}%)"
    )
    console.debug(
        f"- Number of accepted and useful samples: {n_acceptable_useful} ({100*n_acceptable_useful/n_samples:.1f}%)"
    )

    return n_acceptable, n_useful, n_acceptable_useful


def sso_box_sub_compute_candidate_box_measure(
    candidate_box, n_sample, n_useful, measure_function, measure_options, console
):
    """Compute the measure of a candidate box."""
    console.info("Computing candidate box measure... ")
    start_time = time.time()

    measure = measure_function(candidate_box, n_useful / n_sample, **measure_options)

    console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")
    console.debug(f"- Current candidate box measure: {measure}")

    return measure


def sso_box_sub_trimming_operation(
    candidate_box,
    measure,
    design_sample,
    is_acceptable,
    is_useful,
    order_trim,
    trimming_method_function,
    trimming_operation_options,
    console,
):
    """Perform box trimming operation to remove bad design points."""
    console.info("Performing box trimming operation... ")
    start_time = time.time()

    if np.sum(is_acceptable) != len(is_acceptable):
        # If there are bad designs, perform trimming
        label_viable = is_acceptable & is_useful
        candidate_box_trimmed, measure_trimmed = trimming_method_function(
            design_sample,
            label_viable,
            order_trim,
            candidate_box,
            **trimming_operation_options,
        )
    else:
        # No trimming necessary
        candidate_box_trimmed = candidate_box
        measure_trimmed = measure

    console.info(f"Elapsed time is {time.time() - start_time:.3f} seconds.")

    inside_box = is_in_design_box(design_sample, candidate_box_trimmed)
    accepted_inside_box = inside_box & is_acceptable
    useful_inside_box = inside_box & is_useful
    n_sample = len(design_sample)

    removed_count = np.sum(~inside_box)
    console.debug(
        f"- Number of samples removed from candidate box: {removed_count} ({100*removed_count/n_sample:.1f}%)"
    )
    console.debug(
        f"- Number of acceptable samples lost: {np.sum(~inside_box & is_acceptable)}"
    )
    console.debug(f"- Number of useful samples lost: {np.sum(~inside_box & is_useful)}")
    console.debug(
        f"- Number of acceptable samples inside trimmed candidate box: {np.sum(accepted_inside_box)}"
    )
    console.debug(
        f"- Number of useful samples inside trimmed candidate box: {np.sum(useful_inside_box)}"
    )
    console.debug(
        f"- Trimmed candidate box measure: {measure_trimmed} (Relative shrinkage: {-100*(measure_trimmed-measure)/measure:.1f}%)"
    )

    return candidate_box_trimmed, measure_trimmed


# Utility functions that would be imported in a full implementation
class ConsoleLogging:
    """Simple console logging class to mimic MATLAB's logging capabilities."""

    LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "critical": 50}

    def __init__(self, level="info"):
        self.level = self.LEVELS.get(level.lower(), 20)

    def debug(self, message, *args):
        if self.level <= 10:
            print(f"DEBUG: {message % args if args else message}")

    def info(self, message, *args):
        if self.level <= 20:
            print(f"INFO: {message % args if args else message}")

    def warn(self, tag, message):
        if self.level <= 30:
            print(f"WARNING [{tag}]: {message}")

    def error(self, tag, message):
        if self.level <= 40:
            print(f"ERROR [{tag}]: {message}")

    def critical(self, tag, message):
        if self.level <= 50:
            print(f"CRITICAL [{tag}]: {message}")


def design_box_grow_fixed(box, lower_bound, upper_bound, growth_rate):
    """Grow the design box by the given growth rate, bounded by the design space."""
    box_grown = box.copy()

    # Calculate the size of the box in each dimension
    box_size = box[1, :] - box[0, :]

    # Calculate the growth amount in each dimension
    growth_amount = box_size * growth_rate

    # Apply the growth to the upper and lower bounds of the box
    box_grown[0, :] = np.maximum(lower_bound, box[0, :] - growth_amount)
    box_grown[1, :] = np.minimum(upper_bound, box[1, :] + growth_amount)

    return box_grown


def get_current_array_entry(array, index):
    """Get an entry from an array, handling index out of bounds."""
    if index <= len(array):
        return array[index - 1]
    else:
        return array[-1]  # Return the last element if index out of bounds


def design_deficit_to_label_score(deficit):
    """Convert design deficit to labels and scores."""
    is_good = deficit <= 0
    score = -deficit  # Higher score is better
    return is_good, score


def design_requirement_spaces_label(
    requirement_spaces_type, is_good_performance, is_physically_feasible
):
    """Label designs based on requirement spaces type."""
    if requirement_spaces_type.lower() == "solution":
        # Solution spaces: good AND physically feasible designs are acceptable
        is_acceptable = is_good_performance & is_physically_feasible
        is_useful = is_physically_feasible
    elif requirement_spaces_type.lower() == "requirement":
        # Requirement spaces: good designs are acceptable
        is_acceptable = is_good_performance
        is_useful = np.ones_like(is_good_performance)  # All designs are useful
    else:
        # Default to solution spaces
        is_acceptable = is_good_performance & is_physically_feasible
        is_useful = is_physically_feasible

    return is_acceptable, is_useful


def is_in_design_box(design_sample, box):
    """Check if design samples are inside a design box."""
    lower_bound = box[0, :]
    upper_bound = box[1, :]

    is_inside = np.ones(design_sample.shape[0], dtype=bool)

    for i in range(design_sample.shape[1]):
        is_inside = (
            is_inside
            & (design_sample[:, i] >= lower_bound[i])
            & (design_sample[:, i] <= upper_bound[i])
        )

    return is_inside


def trimming_order(is_bad, score, order_preference="score-low-to-high"):
    """Determine the order for trimming based on scores."""
    if not np.any(is_bad):
        return np.array([])

    bad_indices = np.where(is_bad)[0]
    bad_scores = score[bad_indices]

    if order_preference == "score-low-to-high":
        # Sort by score (lowest first)
        sorted_indices = np.argsort(bad_scores)
    else:
        # Sort by score (highest first)
        sorted_indices = np.argsort(-bad_scores)

    return bad_indices[sorted_indices]


def box_trimming_leanness(design_sample, is_useful, trimming_order, box):
    """Apply leanness trimming to a box."""
    box_trimmed = box.copy()

    # If no bad designs or empty box, return original
    if len(trimming_order) == 0 or np.all(box[0, :] >= box[1, :]):
        return box_trimmed

    # Get dimensions of the problem
    n_dim = design_sample.shape[1]

    # Find the minimum box that contains all useful samples
    useful_samples = design_sample[is_useful, :]

    if len(useful_samples) > 0:
        for i in range(n_dim):
            box_trimmed[0, i] = np.min(useful_samples[:, i])
            box_trimmed[1, i] = np.max(useful_samples[:, i])

    return box_trimmed


def sso_stochastic_options(option_type="box"):
    """Return default options for SSO stochastic algorithm."""
    # This would normally import or define a complex options structure
    # For simplicity, we'll return a basic dictionary with required options

    if option_type.lower() == "box":
        return {
            "requirement_spaces_type": "solution",
            "growth_rate": 0.1,
            "max_iter_exploration": 10,
            "max_iter_consolidation": 5,
            "number_samples_per_iteration_exploration": [100],
            "number_samples_per_iteration_consolidation": [100],
            "use_adaptive_growth_rate": True,
            "fix_iter_number_exploration": False,
            "fix_iter_number_consolidation": False,
            "tolerance_measure_change_exploration": 0.01,
            "tolerance_purity_consolidation": 0.95,
            "measure_function": lambda box, purity, **kwargs: np.prod(
                box[1, :] - box[0, :]
            ),
            "measure_options": {},
            "sampling_method_function": lambda box, n_sample, **kwargs: random_sample_in_box(
                box, n_sample
            ),
            "sampling_method_options": {},
            "apply_leanness": "end-only",
            "trimming_order_function": trimming_order,
            "trimming_order_options": {"order_preference": "score-low-to-high"},
            "trimming_operation_function": lambda *args, **kwargs: (
                args[0],
                0,
            ),  # placeholder
            "trimming_operation_options": {},
            "logging_level": "info",
            "minimum_growth_rate": 0.01,
            "maximum_growth_rate": 0.5,
            "growth_adaptation_factor_function": lambda *args, **kwargs: 1.0,  # placeholder
            "growth_adaptation_factor_options": {},
            "minimum_growth_adaptation_factor": 0.5,
            "maximum_growth_adaptation_factor": 2.0,
            "target_accepted_ratio_exploration": 0.5,
            "minimum_growth_purity": 0.0,
            "maximum_growth_purity": 1.0,
        }
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def random_sample_in_box(box, n_sample):
    """Generate random samples within a box."""
    n_dim = box.shape[1]
    lower_bounds = box[0, :]
    upper_bounds = box[1, :]

    # Generate uniform random samples in the box
    samples = np.zeros((n_sample, n_dim))
    for i in range(n_dim):
        samples[:, i] = np.random.uniform(lower_bounds[i], upper_bounds[i], n_sample)

    return samples
