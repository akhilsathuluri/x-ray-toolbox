import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

# This is a standalone class that can be used just for solution space based optimisation independant of the x-ray visualisation tool
# Implements the basic stochastic iteration code for SSO


class XRayOpt:
    def __init__(self, seed=42, log_level=logging.INFO):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=log_level,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing XRayOpt class")
        self.rng: int = np.random.default_rng(seed)
        self.growth_rate: float = 8e-2
        self.max_exploration_iterations: int = 20
        self.max_consolidation_iterations: int = 20
        self.use_adaptive_growth_rate: bool = False
        self.min_purity: float = 1e-3
        self.max_purity: float = 0.999
        self.slack: float = 0.0
        self.apply_leanness: bool = False
        self.sample_size: int = 100
        self.target_accepted_ratio_exploration: float = 0.7
        self.max_growth_rate: float = 0.2
        self.min_growth_rate: float = 0.0
        self.min_growth_rate_adaptation_factor: float = 0.2
        self.max_growth_rate_adaptation_factor: float = 1.5

    def _extract_problem_data(self):
        self.dv_l = self.problem_dv.Lower.to_numpy().astype(np.float64)
        self.dv_u = self.problem_dv.Upper.to_numpy().astype(np.float64)
        self.qoi_l = self.problem_qoi.Lower.to_numpy().astype(np.float64)
        self.qoi_u = self.problem_qoi.Upper.to_numpy().astype(np.float64)
        self.dv_variables = self.problem_dv.Variables.to_list()
        self.qoi_variables = self.problem_qoi.Variables.to_list()
        self.problem_name = self.prob.problem_name

    def _get_problem_info_from_app(self, x_ray_viz_class):
        self.problem_path = x_ray_viz_class.problem_path
        self.prob = x_ray_viz_class.prob
        self.problem_dv = x_ray_viz_class.problem_dv
        self.problem_qoi = x_ray_viz_class.problem_qoi
        self.problem_dv_size = x_ray_viz_class.problem_dv_size
        self.problem_qoi_size = x_ray_viz_class.problem_qoi_size
        self._extract_problem_data()

    def _setup_problem(self, problem_path, problem_class):
        self.problem_path = problem_path
        self.prob = problem_class
        dv = pd.read_csv(self.problem_path + "/input/dv_space.csv", dtype=str)
        qoi = pd.read_csv(self.problem_path + "/input/qoi_space.csv", dtype=str)
        dv.iloc[:, 1:3] = dv.iloc[:, 1:3].astype(np.float64)
        qoi.iloc[:, 1:3] = qoi.iloc[:, 1:3].astype(np.float64)
        self.problem_dv = dv
        self.problem_qoi = qoi
        self.problem_dv_size = dv.shape[0]
        self.problem_qoi_size = qoi.shape[0]
        self._extract_problem_data()

    def update_qoi_bounds(self, qoi):
        self.qoi_l = qoi.Lower.to_numpy().astype(np.float64)
        self.qoi_u = qoi.Upper.to_numpy().astype(np.float64)

    def box_measure_volume(self, dv_box, fraction_useful=1.0):
        volume = np.prod(dv_box[:, 1] - dv_box[:, 0]) * fraction_useful
        return volume

    def dv_box_grow_fixed(self, dv_box):
        dv_box_grown = dv_box + self.growth_rate * (self.dv_u - self.dv_l).reshape(
            -1, 1
        ) * np.array([-1, 1])
        dv_box_grown[:, 0] = np.clip(dv_box_grown[:, 0], self.dv_l, self.dv_u)
        dv_box_grown[:, 1] = np.clip(dv_box_grown[:, 1], self.dv_l, self.dv_u)
        return dv_box_grown

    def compute_qoi_violation(self, qoi_evaluated):
        qoi_violation = np.zeros_like(qoi_evaluated)
        normalisation_factor = 1.0
        qoi_violation_l = (self.qoi_l - qoi_evaluated) / normalisation_factor
        qoi_violation_u = (qoi_evaluated - self.qoi_u) / normalisation_factor
        qoi_violation = np.maximum(qoi_violation_l, qoi_violation_u)
        return qoi_violation

    def compute_qoi_violation_score(self, qoi_violation):
        max_violation = np.max(qoi_violation, axis=1)
        score = np.zeros_like(max_violation)
        feasible_mask = (max_violation < 0).astype(bool)
        score[feasible_mask] = np.mean(qoi_violation[feasible_mask], axis=1)
        score[~feasible_mask] = np.linalg.norm(qoi_violation[~feasible_mask], axis=1)
        return score, feasible_mask

    def get_trimming_order(self, qoi_score):
        positive_indices = np.where(qoi_score > 0)[0]
        sorted_indices = np.argsort(qoi_score[positive_indices])
        sorted_order = positive_indices[sorted_indices]
        trimming_order = np.vstack((sorted_order, np.flip(sorted_order)))
        return trimming_order

    def get_trimmed_box(
        self, dv_box, box_measure, dv_samples, feasible_mask, trimming_order
    ):
        if sum(feasible_mask) == len(feasible_mask):
            self.logger.debug("All samples are feasible, no need to trim")
            return dv_box, box_measure, True
        else:
            dv_box_trimmed, measure_trimmed = self.trim_dv_box(
                dv_samples, feasible_mask, trimming_order, dv_box
            )
        return dv_box_trimmed, measure_trimmed, False

    def dvs_in_box(self, dv_samples, dv_box, dv_violation_score=False):
        lower_bounds = dv_box[:, 0]
        upper_bounds = dv_box[:, 1]
        dv_in_box_mask = np.all(
            (dv_samples >= lower_bounds) & (dv_samples <= upper_bounds), axis=1
        )
        if dv_violation_score:
            dv_violation_score = np.max(
                np.maximum(
                    (dv_box[:, 0] - dv_samples),
                    (dv_samples - dv_box[:, 1]),
                )
                / (dv_box[:, 1] - dv_box[:, 0]),
                axis=1,
            )
            return dv_in_box_mask, dv_violation_score
        return dv_in_box_mask, 0

    def find_closest_viable_point(
        self, dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
    ):
        dvs_in_box_mask, _ = self.dvs_in_box(dv_samples, dv_box)
        if is_lower:
            remain_region = dv_samples[:, dimension] >= dv_samples[trim_idx, dimension]
        else:
            remain_region = dv_samples[:, dimension] <= dv_samples[trim_idx, dimension]

        viables = dv_samples[
            np.logical_and(
                np.logical_and(remain_region, dvs_in_box_mask),
                feasible_mask,
            ),
            dimension,
        ]
        if len(viables) > 0:
            return np.min(viables) if is_lower else np.max(viables)
        else:
            return dv_samples[trim_idx, dimension]

    def calculate_box_quality(self, dv_samples, dv_box, feasible_mask):
        dvs_in_box_mask, _ = self.dvs_in_box(dv_samples, dv_box)
        dvs_in_box_and_feasible = np.logical_and(dvs_in_box_mask, feasible_mask)
        fraction_useful = sum(dvs_in_box_and_feasible) / len(dvs_in_box_mask)
        box_measure = self.box_measure_volume(dv_box, fraction_useful=fraction_useful)
        return box_measure

    def adjust_box_boundary(
        self, dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
    ):
        adjusted_box = dv_box.copy()
        boundary_idx = 0 if is_lower else 1

        closest_viable = dv_samples[trim_idx, dimension]  # Default
        if self.slack < 1:
            closest_viable = self.find_closest_viable_point(
                dv_samples, feasible_mask, trim_idx, dv_box, dimension, is_lower
            )

        # Apply slack factor
        adjusted_box[dimension, boundary_idx] = dv_samples[
            trim_idx, dimension
        ] * self.slack + closest_viable * (1 - self.slack)

        # Calculate box quality
        box_measure = self.calculate_box_quality(
            dv_samples, adjusted_box, feasible_mask
        )

        return adjusted_box, box_measure

    def trim_dv_box(self, dv_samples, feasible_mask, trimming_order, dv_box):
        dv_box_trimmed = dv_box.copy()
        box_measure_trimmed = -np.inf
        dv_box_init = dv_box.copy()
        for ii in range(trimming_order.shape[0]):
            dv_box_current = dv_box_init.copy()
            for jj in range(trimming_order.shape[1]):
                trim_idx = trimming_order[ii, jj]
                dv_box_best = dv_box_current.copy()
                box_measure_best = -np.inf
                for dimension in range(dv_box.shape[0]):
                    for is_lower in [True, False]:
                        adjusted_box, box_measure = self.adjust_box_boundary(
                            dv_samples,
                            feasible_mask,
                            trim_idx,
                            dv_box_current,
                            dimension,
                            is_lower,
                        )
                        if box_measure > box_measure_best:
                            dv_box_best = adjusted_box.copy()
                            box_measure_best = box_measure
                dv_box_current = dv_box_best
            box_measure = self.calculate_box_quality(
                dv_samples, dv_box_current, feasible_mask
            )
            if box_measure > box_measure_trimmed:
                dv_box_trimmed = dv_box_current.copy()
                box_measure_trimmed = box_measure
        return dv_box_trimmed, box_measure_trimmed

    def _set_initial_box(self, dv_box_init=None):
        if dv_box_init is not None:
            self.dv_box_init = dv_box_init
        else:
            self.logger.debug("No initial growth box provided, using midpoint")
            self.dv_box_init = np.column_stack([(self.dv_u + self.dv_l) / 2] * 2)
        self.box_measure_init = self.box_measure_volume(self.dv_box_init)
        self.logger.debug(f"Initial box measure: {self.box_measure_init}")

    def evaluate_and_trim_box(self, dv_box, sample_size, is_exploration=False):
        working_box = dv_box
        if is_exploration:
            logging.debug("Growing candidate box")
            working_box = self.dv_box_grow_fixed(dv_box)
            logging.debug(f"DV box grown: {working_box}")
            logging.debug(f"Current growth rate is: {self.growth_rate}")

        # Generate samples within the design box
        dv_samples = pd.DataFrame(
            self.rng.uniform(
                working_box[:, 0],
                working_box[:, 1],
                size=(sample_size, self.problem_dv_size),
            ),
            columns=self.dv_variables,
        )

        # Evaluate the samples for all the QoIs
        self.prob._compute_commons(dv_samples)
        for method in self.qoi_variables:
            func = getattr(self.prob, method)
            func()

        # Generate labels and scores
        qoi_violation = self.compute_qoi_violation(
            self.prob.var[self.qoi_variables].to_numpy().astype(np.float64)
        )
        qoi_score, feasible_mask = self.compute_qoi_violation_score(qoi_violation)
        if sum(feasible_mask) == 0:
            logging.debug("No feasible samples found, relax constraints and retry")

        # Calculate purity and box measure
        self.purity = sum(feasible_mask) / sample_size
        self.purity = max(min(self.purity, self.max_purity), self.min_purity)
        box_measure_grown = self.box_measure_volume(
            working_box, fraction_useful=self.purity
        )

        if is_exploration:
            logging.debug(f"Purity: {self.purity}")
            if sum(feasible_mask) == 0 or self.purity < self.min_purity:
                logging.debug("Purity is below the threshold, reducing growth rate")

        # Trim the box
        trimming_order = self.get_trimming_order(qoi_score)
        dv_box_trimmed, box_measure_trimmed, convergence_flag = self.get_trimmed_box(
            working_box,
            box_measure_grown,
            dv_samples[self.problem_dv.Variables].to_numpy().astype(np.float64),
            feasible_mask,
            trimming_order,
        )
        logging.debug(f"Box measure after trimming: {box_measure_trimmed}")
        logging.debug(f"Convergence flag after trimming: {convergence_flag}")

        # Apply leanness if requested
        if self.apply_leanness:
            raise NotImplementedError("Leanness is not implemented yet")

        return (
            dv_box_trimmed,
            box_measure_trimmed,
            box_measure_grown,
            convergence_flag,
        )

    def get_growth_adaptation_factor(
        self, purity, measure_increase_fraction_acceptable
    ):
        growth_exponent = (
            self.problem_dv_size
            - (self.problem_dv_size - 1) * measure_increase_fraction_acceptable
        )
        target_purity = self.target_accepted_ratio_exploration
        growth_adaptation_factor = ((1 - target_purity) * purity) / (
            (1 - purity) * target_purity
        ) ** (1 / growth_exponent)
        return growth_adaptation_factor

    def run_sso_stochastic_iteration(self):
        box_measure_trimmed = self.box_measure_init.copy()
        dv_box = self.dv_box_init.copy()
        total_iterations = (
            self.max_exploration_iterations + self.max_consolidation_iterations
        )

        for iteration in range(total_iterations):
            is_exploration_phase = iteration < self.max_exploration_iterations
            phase_name = "Exploration" if is_exploration_phase else "Consolidation"
            phase_iteration = (
                iteration
                if is_exploration_phase
                else iteration - self.max_exploration_iterations
            )

            # logging.debug(f"{phase_name} iteration {phase_iteration}")
            print(f"{phase_name}: iteration {phase_iteration}")

            if is_exploration_phase and iteration > 1 and self.use_adaptive_growth_rate:
                self.purity = max(min(self.purity, self.max_purity), self.min_purity)
                measure_increase = self.box_measure_grown - self.box_measure_prev
                measure_increase_accpetable = max(
                    self.box_measure_grown * self.purity - self.box_measure_prev, 0
                )
                measure_increase_fraction_acceptable = (
                    measure_increase_accpetable / measure_increase
                )
                growth_adaptation_factor = np.clip(
                    self.get_growth_adaptation_factor(
                        self.purity, measure_increase_fraction_acceptable
                    ),
                    self.min_growth_rate_adaptation_factor,
                    self.max_growth_rate_adaptation_factor,
                )
                self.growth_rate = np.clip(
                    self.growth_rate * growth_adaptation_factor,
                    self.min_growth_rate,
                    self.max_growth_rate,
                )
                logging.debug(
                    f"Growth rate adapted to: {self.growth_rate} with factor: {growth_adaptation_factor}"
                )

            if is_exploration_phase:
                logging.debug(
                    f"Iteration {phase_iteration} - Box measure: {box_measure_trimmed}"
                )
            self.box_measure_prev = box_measure_trimmed
            # Evaluate and trim the box
            dv_box, box_measure_trimmed, box_measure_grown, convergence_flag = (
                self.evaluate_and_trim_box(
                    dv_box,
                    self.sample_size,
                    is_exploration=is_exploration_phase,
                )
            )
            self.box_measure_grown = box_measure_grown
            if convergence_flag:
                logging.debug(
                    f"Convergence achieved at iteration {iteration} with box measure: {box_measure_trimmed}"
                )
                break

            # Log phase transition
            if (
                is_exploration_phase
                and iteration == self.max_exploration_iterations - 1
            ):
                logging.debug(
                    f"Exploration phase completed. Box measure: {box_measure_trimmed}"
                )

            if not is_exploration_phase and iteration == total_iterations - 1:
                logging.debug(
                    f"Consolidation phase completed. Box measure: {box_measure_trimmed}"
                )
                logging.debug(f"Final DV box: {dv_box}")

        return dv_box, box_measure_trimmed

    def export_optimisation_result(self, dv_box):
        dv_solution_space = pd.DataFrame(
            {
                "Variables": self.dv_variables,
                "Lower": dv_box[:, 0],
                "Upper": dv_box[:, 1],
                "Units": self.problem_dv["Units"],
                "Description": self.problem_dv["Description"],
            }
        )
        dv_solution_space.to_csv(
            self.problem_path + "/output/dv_solution_space.csv", index=False
        )
        logging.debug(
            f"Exported DV solution space to {self.problem_path}/output/dv_solution_space.csv"
        )
        plot_data = {"sample_size": self.sample_size}
        with open(self.problem_path + "/output/plot_data.json", "w") as f:
            json.dump(plot_data, f)
        logging.debug(
            f"Exported plot data to {self.problem_path}/output/plot_data.json"
        )
        return dv_solution_space


# Example usage

if __name__ == "__main__":
    import sys

    toolbox_path = Path("").resolve().__str__()
    sys.path.append(toolbox_path)
    from problems.CrashDesign.library.CrashDesign import CrashDesign

    ssOpt = XRayOpt(seed=0, log_level=logging.DEBUG)
    ssOpt._setup_problem(
        problem_path=toolbox_path + "/problems/CrashDesign/",
        problem_name="Crash Design example",
        problem_class=CrashDesign(),
    )
    box_init_guess = np.column_stack([[4.05e5, 2e3, 4.05e5, 15.6, 0.3, 0.3]] * 2)
    ssOpt._set_initial_box(box_init_guess)
    dv_box, box_measure = ssOpt.run_sso_stochastic_iteration()
    print(f"Final box measure: {box_measure}")
    print(f"Final design variable box: {dv_box}")

    ssOpt.export_optimisation_result(dv_box)
