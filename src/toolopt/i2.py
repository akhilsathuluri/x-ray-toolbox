import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Union

# --- Assume the following external dependencies exist and work as described ---

# Placeholder for a Console Logging class similar to MATLAB's ConsoleLogging
class ConsoleLogging:
    """
    Placeholder for a console logging class.
    Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    def __init__(self, level: str = 'INFO'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg: str, *args):
        self.logger.info(msg, *args)

    def debug(self, msg: str, *args):
        self.logger.debug(msg, *args)

    def warning(self, msg: str, *args):
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args):
        self.logger.error(msg, *args)
        # In MATLAB, console.error might throw an exception. We will raise here.
        # raise RuntimeError(msg % args) # Decided not to raise by default, just log

    def critical(self, msg: str, *args):
        self.logger.critical(msg, *args)
        # raise RuntimeError(msg % args) # Decided not to raise by default, just log


# Placeholder for a DesignEvaluatorBase class
class DesignEvaluatorBase:
    """
    Placeholder base class for evaluating design sample points.
    Assumed to have an evaluate method.
    """
    def evaluate(self, design_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Evaluates design samples.

        Args:
            design_samples: (n_sample, n_design_variable) array of design points.

        Returns:
            performance_deficit: (n_sample, 1) array of performance deficits (lower is better, <= 0 is good).
            physical_feasibility_deficit: (n_sample, 1) array of physical feasibility deficits (lower is better, <= 0 is feasible), or empty array if not applicable.
            evaluation_output: Any other output from the evaluation.
        """
        # Mock implementation: random deficits
        n_sample, n_dim = design_samples.shape
        performance_deficit = np.random.rand(n_sample, 1) - 0.5 # 50% good
        physical_feasibility_deficit = np.random.rand(n_sample, 1) - 0.2 # 80% feasible
        evaluation_output = {'some_metric': np.random.rand(n_sample, 1)}
        return performance_deficit, physical_feasibility_deficit, evaluation_output

# Placeholder for sso_stochastic_options function
def sso_stochastic_options(opt_type: str) -> Dict[str, Any]:
    """
    Placeholder function to get default stochastic optimization options.
    """
    options = {
        'RequirementSpacesType': 'feasible_and_good', # or 'feasible', 'good'
        'ApplyLeanness': 'end-only', # 'always', 'end-only', 'never'
        'MeasureFunction': lambda box, purity, *args: np.prod(box[1, :] - box[0, :]) * purity if box.shape[0] == 2 else 0, # Default measure is volume * purity
        'MeasureOptions': {},
        'TrimmingOperationOptions': {},
        'LoggingLevel': 'INFO',
        'MaxIterExploration': 10,
        'GrowthRate': 0.1,
        'UseAdaptiveGrowthRate': True,
        'MaximumGrowthPurity': 0.9,
        'MinimumGrowthPurity': 0.1,
        'TargetAcceptedRatioExploration': 0.5,
        'GrowthAdaptationFactorFunction': lambda purity, target, dim, frac_increase, *args: 1.0 + 2.0 * (purity - target), # Simple adaptation
        'GrowthAdaptationFactorOptions': {},
        'MaximumGrowthAdaptationFactor': 2.0,
        'MinimumGrowthAdaptationFactor': 0.5,
        'FixIterNumberExploration': False,
        'ToleranceMeasureChangeExploration': 1e-3,
        'NumberSamplesPerIterationExploration': 100,
        'MaxIterConsolidation': 20,
        'FixIterNumberConsolidation': False,
        'TolerancePurityConsolidation': 0.95,
        'NumberSamplesPerIterationConsolidation': 500,
        'SamplingMethodFunction': lambda box, n, *args: np.random.uniform(box[0, :], box[1, :], (n, box.shape[1])), # Default is uniform sampling
        'SamplingMethodOptions': {},
        'TrimmingOrderFunction': lambda is_bad, score, *args: np.argsort(score[is_bad].flatten()), # Default order by score
        'TrimmingOrderOptions': {},
        'TrimmingOperationFunction': lambda samples, is_viable, order, box, *args: (box, np.prod(box[1, :] - box[0, :])), # Placeholder trimming
    }
    return options

# Placeholder for parser_variable_input_to_structure
def parser_variable_input_to_structure(*args) -> Dict[str, Any]:
    """
    Placeholder function to parse variable input arguments into a dictionary.
    Assumes input is pairs of name, value.
    """
    input_options = {}
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            input_options[args[i]] = args[i+1]
    return input_options

# Placeholder for merge_name_value_pair_argument
def merge_name_value_pair_argument(defaults: Union[Dict, List], overrides: Dict) -> Union[Dict, List]:
    """
    Placeholder function to merge default options with overrides.
    Can handle dictionaries or lists of key-value pairs.
    """
    if isinstance(defaults, list):
        default_dict = dict(defaults)
        merged = {**default_dict, **overrides}
        return list(merged.items())
    elif isinstance(defaults, dict):
         return {**defaults, **overrides}
    else:
        return overrides # Should not happen with expected inputs


# Placeholder for design_box_grow_fixed
def design_box_grow_fixed(candidate_box: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, growth_rate: float) -> np.ndarray:
    """
    Placeholder function to grow a candidate box by a given rate,
    respecting lower and upper bounds and fixed dimensions.
    Assumes fixed dimensions are where candidate_box[0, i] == candidate_box[1, i].
    """
    grown_box = candidate_box.copy()
    n_dim = grown_box.shape[1]

    for i in range(n_dim):
        if grown_box[0, i] != grown_box[1, i]: # Not a fixed dimension
            current_range = grown_box[1, i] - grown_box[0, i]
            grow_amount = current_range * growth_rate / 2.0
            grown_box[0, i] = max(lower_bound[0, i], grown_box[0, i] - grow_amount)
            grown_box[1, i] = min(upper_bound[0, i], grown_box[1, i] + grow_amount)
        # Else: fixed dimension, do not grow

    return grown_box


# Placeholder for get_current_array_entry
def get_current_array_entry(arr: Union[List, np.ndarray, int], iteration: int) -> int:
    """
    Placeholder function to get the current value from an array or scalar
    based on the iteration number (1-based).
    If arr is scalar, return scalar.
    If arr is array, return arr[min(iteration-1, len(arr)-1)].
    """
    if isinstance(arr, (int, float)):
        return int(arr)
    elif isinstance(arr, (list, np.ndarray)):
        return int(arr[min(iteration - 1, len(arr) - 1)])
    else:
        raise TypeError("Input 'arr' must be scalar, list, or numpy array.")

# Placeholder for design_deficit_to_label_score
def design_deficit_to_label_score(deficit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder function to convert deficit values to boolean labels and scores.
    Assumes deficit <= 0 is good/feasible.
    Score is often related to the deficit, e.g., -deficit.
    """
    label = deficit.flatten() <= 0
    score = -deficit.flatten() # Simple score: higher is better
    return label, score

# Placeholder for design_requirement_spaces_label
def design_requirement_spaces_label(req_type: str, is_good_performance: np.ndarray, is_physically_feasible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder function to label samples based on requirement spaces type.
    req_type: 'feasible_and_good', 'feasible', 'good'
    is_acceptable: samples that meet the acceptable criteria
    is_useful: samples that are useful for finding the space (often is_acceptable)
    """
    if req_type == 'feasible_and_good':
        is_acceptable = is_good_performance & is_physically_feasible
        is_useful = is_acceptable # Useful points are those inside the acceptable space
    elif req_type == 'feasible':
        is_acceptable = is_physically_feasible
        is_useful = is_good_performance & is_physically_feasible # Useful points are feasible AND good
    elif req_type == 'good':
        is_acceptable = is_good_performance
        is_useful = is_good_performance & is_physically_feasible # Useful points are good AND feasible
    else:
        raise ValueError(f"Unknown RequirementSpacesType: {req_type}")

    return is_acceptable, is_useful

# Placeholder for trimming_order
def trimming_order(is_bad: np.ndarray, score: np.ndarray, OrderPreference: str = 'score-low-to-high') -> np.ndarray:
    """
    Placeholder function to determine the order of trimming based on score.
    Assumes 'score-low-to-high' means trim points with lower scores first among 'bad' points.
    Returns indices relative to the *original* sample array.
    """
    if OrderPreference == 'score-low-to-high':
        # Get indices of bad points
        bad_indices = np.where(is_bad)[0]
        # Get scores of bad points
        bad_scores = score[is_bad]
        # Get the sorted indices of the bad scores
        sorted_bad_indices = np.argsort(bad_scores)
        # Return the original indices in the desired trimming order
        return bad_indices[sorted_bad_indices]
    else:
        # Default: just trim the bad points in their original order
        return np.where(is_bad)[0]


# Placeholder for box_trimming_leanness
def box_trimming_leanness(design_samples: np.ndarray, is_useful: np.ndarray, trimming_order: np.ndarray, candidate_box: np.ndarray) -> np.ndarray:
    """
    Placeholder function to perform trimming for leanness.
    This is a complex operation that would require specific implementation details.
    A simple mock might just return the input box or a slightly smaller one.
    """
    # This is a highly simplified mock. Real implementation would iterate through
    # points to trim based on the order and check if removing a point
    # increases the "leanness" (e.g., by shrinking the box without losing useful points).
    # For this translation, we'll just return the box as is, implying no leanness trimming
    # occurs in this mock, or return a slightly smaller box for demonstration.

    # Example of returning a slightly smaller box (not a real leanness trim)
    # trimmed_box = candidate_box.copy()
    # shrink_factor = 0.01
    # box_range = trimmed_box[1, :] - trimmed_box[0, :]
    # trimmed_box[0, :] += box_range * shrink_factor / 2
    # trimmed_box[1, :] -= box_range * shrink_factor / 2
    # trimmed_box = np.maximum(trimmed_box, candidate_box[0,:]) # Ensure no inverted box
    # trimmed_box = np.minimum(trimmed_box, candidate_box[1,:])

    return candidate_box # Default behavior: no leanness trimming in this mock

# Placeholder for sso_box_sub_trimming_operation
def sso_box_sub_trimming_operation(
    candidate_box: np.ndarray,
    measure: float,
    design_sample: np.ndarray,
    is_acceptable: np.ndarray,
    is_useful: np.ndarray,
    order_trim: np.ndarray,
    trimming_method_function: callable,
    trimming_operation_options: Dict,
    console: ConsoleLogging
) -> Tuple[np.ndarray, float]:
    """
    Placeholder for the trimming operation.
    Calls the specified trimming_method_function.
    """
    console.info('Performing box trimming operation... ')
    start_time = time.perf_counter()

    # The actual trimming_method_function needs to be implemented based on the algorithm.
    # A typical box trimming might try removing points and checking if the box containing
    # the remaining viable points gets smaller while still containing a high percentage
    # of the useful points.
    # The provided MATLAB code structure suggests trimming_method_function takes:
    # (design_sample, label_viable, order_trim, candidate_box, *trimming_operation_options)

    label_viable = is_acceptable & is_useful

    # If there are points to trim (points that are not acceptable)
    if np.sum(is_acceptable) != len(is_acceptable):
        # Call the actual trimming function provided in options
        # Note: The signature of trimming_method_function is assumed based on the MATLAB usage.
        # The mock `trimming_method_function` above is too simple; a real one would use samples, labels, and order.
        # Let's refine the mock `trimming_method_function` assumption slightly for better consistency with usage here.
        # Assume trimming_method_function: (samples, is_viable, order_trim, current_box, options) -> new_box
        # And measure is re-calculated afterwards based on the new box.
        # This is still a guess based on the limited code. A real implementation is needed.
        # For now, let's make a slightly less naive mock for trimming_method_function
        # that might shrink the box if there are many unacceptable points near the boundary.
        # This is just illustrative and not the actual algorithm logic.

        # A more plausible (but still generic) mock for trimming_method_function:
        # It finds the bounds of the 'viable' points and sets the trimmed box to these bounds.
        if np.sum(label_viable) > 0:
             viable_points = design_sample[label_viable, :]
             trimmed_box_min = np.min(viable_points, axis=0)
             trimmed_box_max = np.max(viable_points, axis=0)
             candidate_box_trimmed = np.vstack((trimmed_box_min, trimmed_box_max))
        else:
             # If no viable points, maybe return a zero-volume box or the original?
             # Returning the original might prevent getting stuck if sampling was unlucky.
             # Let's return a point if there are any acceptable/useful points, otherwise original.
             if np.sum(is_acceptable) > 0:
                  candidate_box_trimmed = np.vstack((design_sample[is_acceptable,:][0,:], design_sample[is_acceptable,:][0,:])) # Just pick the first acceptable point
             elif np.sum(is_useful) > 0:
                   candidate_box_trimmed = np.vstack((design_sample[is_useful,:][0,:], design_sample[is_useful,:][0,:])) # Just pick the first useful point
             else:
                  candidate_box_trimmed = candidate_box # Fallback if truly no good/useful points

        # Recalculate measure for the trimmed box
        # The measure function in the MATLAB code is `measureFunction(candidateBox, nUseful/nSample, measureOptions{:})`.
        # It depends on the *purity* within the *grown* box, not the trimmed box directly?
        # This seems slightly inconsistent or implies the measure function knows how to handle this context.
        # Let's assume the measure needs to be re-computed based on how many *original* samples
        # fall into the *newly trimmed* box, and the purity *within that new box*.

        inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
        n_useful_inside_trimmed = np.sum(inside_box_new & is_useful)
        n_inside_trimmed = np.sum(inside_box_new)

        purity_inside_trimmed = n_useful_inside_trimmed / n_inside_trimmed if n_inside_trimmed > 0 else 0

        # Re-compute measure based on the trimmed box and the points inside it
        # Assuming the measure function signature is (box, purity_in_box, *options)
        measure_trimmed = trimming_method_function(design_sample, label_viable, order_trim, candidate_box, trimming_operation_options).measure # Assuming the trimming_method_function *returns* the new box and its measure


    else:
        # No trimming necessary if all points are acceptable
        candidate_box_trimmed = candidate_box
        measure_trimmed = measure # Measure doesn't change if box doesn't trim

    end_time = time.perf_counter()
    console.info('Elapsed time is %g seconds.\n', end_time - start_time)

    # Log trimming results
    inside_box_after_trim = is_in_design_box(design_sample, candidate_box_trimmed)
    accepted_inside_box = inside_box_after_trim & is_acceptable
    useful_inside_box = inside_box_after_trim & is_useful
    n_sample = design_sample.shape[0]

    console.debug('- Number of samples removed from candidate box: %g (%g%%)\n',
                  n_sample - np.sum(inside_box_after_trim), 100 * (n_sample - np.sum(inside_box_after_trim)) / n_sample)
    console.debug('- Number of acceptable samples lost: %g\n', np.sum(~inside_box_after_trim & is_acceptable))
    console.debug('- Number of useful samples lost: %g\n', np.sum(~inside_box_after_trim & is_useful))
    console.debug('- Number of acceptable samples inside trimmed candidate box: %g\n', np.sum(accepted_inside_box))
    console.debug('- Number of useful samples inside trimmed candidate box: %g\n', np.sum(useful_inside_box))

    original_measure = measure # The measure *before* this trim operation (measureGrown or measure in Phase II)
    relative_shrinkage = -100 * (measure_trimmed - original_measure) / original_measure if original_measure != 0 else 0

    console.debug('- Trimmed candidate box measure: %g (Relative shrinkage: %g%%)\n', measure_trimmed, relative_shrinkage)

    return candidate_box_trimmed, measure_trimmed

# Placeholder for is_in_design_box
def is_in_design_box(design_samples: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Placeholder function to check if design samples are within a box.
    Box is [lower_bounds; upper_bounds].
    """
    if box.shape[0] != 2:
         raise ValueError("Box must have 2 rows (lower and upper bounds).")
    lower_bounds = box[0, :]
    upper_bounds = box[1, :]
    # Check if each point is greater than or equal to lower bounds AND
    # less than or equal to upper bounds for all dimensions.
    is_inside = np.all(design_samples >= lower_bounds, axis=1) & np.all(design_samples <= upper_bounds, axis=1)
    return is_inside.reshape(-1, 1) # Return as column vector

# --- End of assumed external dependencies ---


# Helper function translations

def sso_box_sub_generate_new_sample_points(
    candidate_box: np.ndarray,
    n_sample: int,
    sampling_function: callable,
    sampling_options: Dict,
    console: ConsoleLogging
) -> np.ndarray:
    """
    Generate new sample points within the candidate box.
    """
    console.info('Generating new sample points in candidate box... ')
    start_time = time.perf_counter()

    design_sample = sampling_function(candidate_box, n_sample, **sampling_options)

    end_time = time.perf_counter()
    console.info('Elapsed time is %g seconds.\n', end_time - start_time)
    console.debug('- Number of samples generated: %g\n', n_sample)

    return design_sample


def sso_box_sub_evaluate_sample_points(
    design_evaluator: DesignEvaluatorBase,
    design_sample: np.ndarray,
    console: ConsoleLogging
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Evaluate design sample points using the provided evaluator.
    """
    console.info('Evaluating sample points... ')
    start_time = time.perf_counter()

    performance_deficit, physical_feasibility_deficit, output_evaluation = design_evaluator.evaluate(design_sample)

    is_good_performance, score = design_deficit_to_label_score(performance_deficit)

    if physical_feasibility_deficit is not None and physical_feasibility_deficit.size > 0:
        is_physically_feasible, _ = design_deficit_to_label_score(physical_feasibility_deficit)
    else:
        is_physically_feasible = np.ones((design_sample.shape[0],), dtype=bool)

    end_time = time.perf_counter()
    console.info('Elapsed time is %g seconds.\n', end_time - start_time)

    n_samples = design_sample.shape[0]
    console.debug('- Number of good samples: %g (%g%%)\n',
                  np.sum(is_good_performance), 100 * np.sum(is_good_performance) / n_samples if n_samples > 0 else 0)
    console.debug('- Number of bad samples: %g (%g%%)\n',
                  np.sum(~is_good_performance), 100 * np.sum(~is_good_performance) / n_samples if n_samples > 0 else 0)
    console.debug('- Number of physically feasible samples: %g (%g%%)\n',
                  np.sum(is_physically_feasible), 100 * np.sum(is_physically_feasible) / n_samples if n_samples > 0 else 0)
    console.debug('- Number of physically infeasible samples: %g (%g%%)\n',
                  np.sum(~is_physically_feasible), 100 * np.sum(~is_physically_feasible) / n_samples if n_samples > 0 else 0)

    return is_good_performance, is_physically_feasible, score, output_evaluation


def sso_box_sub_label_samples_requirement_spaces(
    requirement_spaces_type: str,
    is_good_performance: np.ndarray,
    is_physically_feasible: np.ndarray,
    console: ConsoleLogging
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label samples according to the desired requirement spaces problem type.
    """
    console.info('Creating labels for each design... ')
    start_time = time.perf_counter()

    is_acceptable, is_useful = design_requirement_spaces_label(
        requirement_spaces_type, is_good_performance, is_physically_feasible)

    end_time = time.perf_counter()
    console.info('Elapsed time is %g seconds.\n', end_time - start_time)

    n_sample = len(is_good_performance)
    console.debug('- Number of accepted samples: %g (%g%%)\n',
                  np.sum(is_acceptable), 100 * np.sum(is_acceptable) / n_sample if n_sample > 0 else 0)
    console.debug('- Number of rejected samples: %g (%g%%)\n',
                  np.sum(~is_acceptable), 100 * np.sum(~is_acceptable) / n_sample if n_sample > 0 else 0)
    console.debug('- Number of useful samples: %g (%g%%)\n',
                  np.sum(is_useful), 100 * np.sum(is_useful) / n_sample if n_sample > 0 else 0)
    console.debug('- Number of useless samples: %g (%g%%)\n',
                  np.sum(~is_useful), 100 * np.sum(~is_useful) / n_sample if n_sample > 0 else 0)

    return is_acceptable, is_useful


def sso_box_sub_count_label_acceptable_useful(
    is_acceptable: np.ndarray,
    is_useful: np.ndarray,
    console: ConsoleLogging
) -> Tuple[int, int, int]:
    """
    Count the number of acceptable, useful, and acceptable AND useful samples.
    """
    n_acceptable = np.sum(is_acceptable)
    n_useful = np.sum(is_useful)
    n_acceptable_useful = np.sum(is_acceptable & is_useful)

    n_samples = len(is_acceptable)
    console.debug('- Number of accepted samples: %g (%g%%)\n',
                  n_acceptable, 100 * n_acceptable / n_samples if n_samples > 0 else 0)
    console.debug('- Number of useful samples: %g (%g%%)\n',
                  n_useful, 100 * n_useful / n_samples if n_samples > 0 else 0)
    console.debug('- Number of accepted and useful samples: %g (%g%%)\n',
                  n_acceptable_useful, 100 * n_acceptable_useful / n_samples if n_samples > 0 else 0)

    return int(n_acceptable), int(n_useful), int(n_acceptable_useful)


def sso_box_sub_compute_candidate_box_measure(
    candidate_box: np.ndarray,
    n_sample: int,
    n_useful: int,
    measure_function: callable,
    measure_options: Dict,
    console: ConsoleLogging
) -> float:
    """
    Compute the measure of the candidate box.
    """
    console.info('Computing candidate box measure... ')
    start_time = time.perf_counter()

    # Handle potential division by zero if no samples were taken
    purity = n_useful / n_sample if n_sample > 0 else 0

    # Assume measure_function signature is (box, purity, *options)
    measure = measure_function(candidate_box, purity, **measure_options)

    end_time = time.perf_counter()
    console.info('Elapsed time is %g seconds.\n', end_time - start_time)
    console.debug('- Current candidate box measure: %g\n', measure)

    return float(measure)

# Main function translation
def sso_box_stochastic(
    design_evaluator: DesignEvaluatorBase,
    initial_box: np.ndarray,
    design_space_lower_bound: np.ndarray,
    design_space_upper_bound: np.ndarray,
    *args, **kwargs
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    SSO_BOX_STOCHASTIC Box-shaped solution spaces optimization (Stochastic method)
    SSO_BOX_STOCHASTIC uses a modified version of the stochastic method to
    compute optimal soluton (or requirement) spaces.

    CANDIDATEBOX = SSO_BOX_STOCHASTIC(DESIGNEVALUATOR,INITIALBOX,
    DESIGNSPACELOWERBOUND,DESIGNSPACEUPPERBOUND) starts the algorithm in
    INITIALBOX and finds the optimum box-shaped solution (or requirement) spaces
    within the design space defined by DESIGNSPACELOWERBOUND and
    DESIGNSPACEUPPERBOUND, evaluating design sample points with DESIGNEVALUATOR
    and returning the optimal box CANDIDATEBOX.

    CANDIDATEBOX = SSO_BOX_STOCHASTIC(DESIGNEVALUATOR,INITIALBOX,
    DESIGNSPACELOWERBOUND,DESIGNSPACEUPPERBOUND,OPTIONS) also allows one to
    change the options being used through a dictionary OPTIONS.

    [CANDIDATEBOX,PROBLEMDATA] = SSO_BOX_STOCHASTIC(...) additionally returns
    the fixed problem data in PROBLEMDATA.

    [CANDIDATEBOX,PROBLEMDATA,ITERATIONDATA] = SSO_BOX_STOCHASTIC(...)
    additionally returns data generated at each iteration of the process
    ITERATIONDATA.

    Args:
        design_evaluator: An instance of DesignEvaluatorBase.
        initial_box: (1, n_design_variable) or (2, n_design_variable) array.
        design_space_lower_bound: (1, n_design_variable) array.
        design_space_upper_bound: (1, n_design_variable) array.
        *args, **kwargs: Variable input arguments for options.

    Returns:
        candidate_box: (2, n_design_variable) array - The optimal box.
        problem_data: dict - Fixed problem data.
        iteration_data: list of dict - Data from each iteration.
    """
    # Options
    default_options = sso_stochastic_options('box')
    # Combine positional args and keyword args into a single dictionary for options parsing
    input_options_dict = parser_variable_input_to_structure(*args)
    input_options_dict.update(kwargs)
    options = merge_name_value_pair_argument(default_options, input_options_dict)

    # extract options as necessary
    requirement_spaces_type = options['RequirementSpacesType']
    apply_leanness_each_trim = options['ApplyLeanness'].lower() == 'always'
    apply_leanness_final_trim = options['ApplyLeanness'].lower() in ['always', 'end-only']

    # trimming options (assuming TrimmingOperationOptions in default options
    # is a dict that needs to be merged with specific measure options if provided)
    # Re-parsing based on the MATLAB code's merge logic:
    # [~,trimmingOperationOptions] = merge_name_value_pair_argument({'MeasureFunction',options.MeasureFunction,'MeasureOptions',options.MeasureOptions}, options.TrimmingOperationOptions);
    # This is a bit unusual, merging measure function/options INTO trimming operation options.
    # Let's assume trimmingOperationOptions dict can contain 'MeasureFunction' and 'MeasureOptions' overrides.
    trimming_operation_options = merge_name_value_pair_argument(
        {'MeasureFunction': options['MeasureFunction'], 'MeasureOptions': options['MeasureOptions']},
        options.get('TrimmingOperationOptions', {}) # Use .get for safety
    )
    # Ensure MeasureOptions is a dict if it came from the merge
    if 'MeasureOptions' in trimming_operation_options and not isinstance(trimming_operation_options['MeasureOptions'], dict):
         trimming_operation_options['MeasureOptions'] = dict(trimming_operation_options['MeasureOptions']) # Convert from list if necessary


    # logging verbosity
    console = ConsoleLogging(options['LoggingLevel'])

    # Initial Candidate Box
    if initial_box.shape[0] == 1:
        candidate_box = np.vstack((initial_box, initial_box))  # single point
    elif initial_box.shape[0] == 2:
        candidate_box = initial_box  # candidate box
    else:
        console.error('SSOBoxOptStochastic:InitialGuessWrong Error. Initial guess incompatible in ''sso_box_stochastic''.')
        # In MATLAB, this would stop execution or raise an error.
        # We've logged an error, can also raise an exception if desired.
        # raise ValueError("Initial guess incompatible. Must be (1, N) or (2, N).")
        return np.array([]), {}, [] # Return empty if error and not raising

    # Initial Measure (of the initial_box, before any growth/trimming)
    # Need to evaluate the initial box? The code computes measureTrimmed based on initialBox.
    # This seems to assume the initial box is the *trimmed* box from a conceptual previous step.
    # Let's calculate the measure assuming no samples yet, maybe using the volume of the initial box?
    # The measure function requires purity. Initial purity is unknown.
    # The MATLAB code initializes measureTrimmed = 0 if inf or nan. This suggests 0 is a valid starting point.
    # Let's initialize measure_trimmed based on the initial box's volume, assuming purity 1 for the start.
    measure_trimmed = options['MeasureFunction'](candidate_box, 1.0, **options['MeasureOptions'])
    if np.isinf(measure_trimmed) or np.isnan(measure_trimmed):
         measure_trimmed = 0

    n_dimension = design_space_lower_bound.shape[1]

    # Log Initialization
    problem_data: Dict[str, Any] = {}
    iteration_data: List[Dict[str, Any]] = []
    is_output_problem_data = True # Always return problem_data as per function signature
    is_output_iteration_data = True # Always return iteration_data as per function signature

    if is_output_problem_data:
        problem_data = {
            'DesignEvaluator': design_evaluator,
            'InitialBox': candidate_box.copy(),
            'DesignSpaceLowerBound': design_space_lower_bound.copy(),
            'DesignSpaceUpperBound': design_space_upper_bound.copy(),
            'Options': options,
            # Using numpy.random.get_state() as an equivalent to MATLAB's rng
            'InitialRNGState': np.random.get_state(),
        }

    # Initial state for logging iteration data
    log_index = 0 # Use 0-based indexing for Python lists

    # Initialize previous measure for convergence check
    measure_previous = measure_trimmed

    # Phase I - Exploration
    i_exploration = 1
    growth_rate = options['GrowthRate']
    has_converged_exploration = False

    console.info("Starting Phase I - Exploration...")

    while ((not has_converged_exploration) and (i_exploration <= options['MaxIterExploration'])):
        console.info('=' * 120)
        console.info('Initiating Phase I - Exploration: Iteration %d\n', i_exploration)

        # Modification Step B - Growth: Extend Candidate Box
        # Change growth rate depending on previous result (only after first iteration)
        if i_exploration > 1 and options['UseAdaptiveGrowthRate']:
            console.info('Adapting growth rate... ')
            tic_adapt = time.perf_counter()

            # Purity and measure increase from the *previous* iteration's results
            # Need nAcceptable and nSample from the *previous* iteration.
            # Let's assume the data from the previous iteration's logging is available.
            # This implies we need to access the last entry of iteration_data,
            # which is tricky if we only append *after* the convergence check.
            # A better approach is to store these values explicitly for the adaptive step.

            # Let's use the nAcceptable and nSample from the *last completed* iteration (i.e., i_exploration - 1).
            # If i_exploration == 1, this block is skipped.
            if log_index > 0: # Check if there's data from the previous iteration
                 prev_iter_data = iteration_data[log_index - 1]
                 prev_n_acceptable = np.sum(prev_iter_data['IsAcceptable'])
                 prev_n_sample = len(prev_iter_data['IsAcceptable']) # Assuming IsAcceptable is logged

                 if prev_n_sample > 0:
                     purity = prev_n_acceptable / prev_n_sample
                 else:
                     purity = 0.0 # Avoid division by zero

                 purity = max(min(purity, options['MaximumGrowthPurity']), options['MinimumGrowthPurity'])

                 # measureGrown and measurePrevious are from the *current* iteration's perspective,
                 # comparing the grown box measure to the trimmed box measure *before* growth.
                 # The adaptive step uses the outcome of the *previous* iteration's growth and trim.
                 # This part of the adaptive logic in the MATLAB code (`increaseMeasure = measureGrown - measurePrevious;`)
                 # seems to refer to the measure change in the current iteration before trim vs previous iteration's trimmed measure.
                 # The adaptive factor calculation then uses purity from the current sample, which is evaluated *after* growth.
                 # This suggests the adaptive logic *uses* the results of the current iteration's sampling/evaluation.

                 # So, we need the purity from the *current* iteration's sampling *before* this adaptive step runs.
                 # This implies the adaptive step should happen *after* evaluation and labeling.
                 # Re-reading the MATLAB code, the adaptive growth rate calculation *is* indeed done *after* the evaluation/labeling/counting
                 # in the previous iteration, before the growth step of the current iteration.
                 # So, the variable names `nAcceptable` and `nSample` within the adaptive block
                 # refer to the values computed in the *immediately preceding* iteration (i_exploration - 1).

                 # Let's assume these values (nAcceptable, nSample, measureGrown, measurePrevious) are available from the prior step.
                 # However, the structure of the loop in the MATLAB code calculates `measureGrown`
                 # *after* the adaptive step and *after* sampling.
                 # This creates a dependency on future values or implies `measureGrown` in the adaptive step
                 # refers to the measure of the box *before* trimming in the *previous* iteration.

                 # Let's follow the MATLAB code's apparent structure closely. The adaptive logic
                 # uses `purity = nAcceptable/nSample;`. These variables are computed
                 # *after* sampling and evaluation *within the current loop iteration*.
                 # This means the adaptive rate for iteration `i_exploration` is based on the results of iteration `i_exploration-1`.
                 # The MATLAB code places the adaptive step *before* the growth and sampling of the current iteration.
                 # This implies `nAcceptable` and `nSample` must be holding values from the previous iteration.

                 # To reconcile this, let's compute and store `nAcceptable`, `nSample`, and `measureTrimmed` at the end of the loop
                 # and use them at the beginning of the *next* iteration's adaptive step.
                 # Let's call the variables holding previous iteration's results `prev_n_acceptable`, `prev_n_sample`, `prev_measure_trimmed`.

                 # This block is executed at the start of iteration `i_exploration`, using results from `i_exploration - 1`.
                 # We need `prev_n_acceptable`, `prev_n_sample`, `prev_measure_trimmed` which were the results *after* trimming
                 # in the previous iteration.

                 # Initializing variables for the adaptive step for the first iteration:
                 if i_exploration == 1:
                      prev_n_acceptable = 0 # No acceptable samples before the first iteration
                      prev_n_sample = 0 # No samples before the first iteration
                      prev_measure_trimmed = measure_trimmed # Initial measure

                 # Now, use these previous values for the adaptive step in the current iteration
                 # Note: The original MATLAB code used `measureGrown - measurePrevious` here.
                 # `measurePrevious` is the measure *before* growth in the *current* iteration (which is `prev_measure_trimmed`).
                 # `measureGrown` is computed *after* growth in the *current* iteration.
                 # This still indicates the adaptive step happens *after* growth and sampling, which contradicts its placement.

                 # Let's assume the adaptive logic *intends* to use the results of the *previous* iteration
                 # (purity and measure change achieved by growing and trimming in the previous step).
                 # In the previous iteration (i_exploration - 1), we had a `candidateBoxTrimmed` and `measureTrimmed`.
                 # We also had a `candidateBoxGrown` and `measureGrown` *before* trimming.
                 # The measure change *achieved* in the previous step was `measureTrimmed_prev - measureTrimmed_before_trim_prev`.
                 # No, the adaptive step calculates `increaseMeasure = measureGrown - measurePrevious`.
                 # `measurePrevious` = `measureTrimmed` from the end of the previous iteration.
                 # `measureGrown` = measure of the box *after* growth in the *current* iteration, *before* trimming.
                 # This means the adaptive step *must* happen *after* the current iteration's box growth.

                 # Okay, let's adjust the flow to match the MATLAB code's placement of the adaptive step.
                 # The adaptive step *is* before the growth and sampling. This implies the variables used (`purity`, `measureGrown`, `measurePrevious`)
                 # must refer to values from the *end* of the previous iteration's process (after trimming).
                 # `measurePrevious` is correctly initialized outside the loop to the initial measure, and updated at the end of the loop.
                 # `measureGrown` is calculated *after* growth in the current iteration, but it's used in the adaptive step *before* growth.
                 # This is confusing. Let's assume `measureGrown` in the adaptive step is a typo and it should be `measureTrimmed` from the previous iteration?
                 # Or maybe it's a look-ahead where `measureGrown` refers to the measure of the box *about to be grown* in the current step,
                 # and `measurePrevious` is the measure *before* the growth in the previous step. No, that doesn't fit either.

                 # Let's strictly follow the variable names used: `measureGrown` is computed after growing the box *in the current iteration*.
                 # `measurePrevious` is the `measureTrimmed` from the *previous* iteration.
                 # The adaptive step *must* therefore happen after the current iteration's box has been grown and its measure calculated.
                 # This contradicts the placement in the MATLAB code.

                 # **Assumption:** The MATLAB code has the adaptive step placed slightly incorrectly logically, or `measureGrown` in the adaptive step refers to the measure of the box *before* trimming in the *previous* iteration. Let's assume the latter for now, as it makes more sense algorithmically.
                 # Let `prev_measure_grown` be the measure of the box *before* trimming in the previous iteration.
                 # Let `prev_measure_trimmed` be the measure of the box *after* trimming in the previous iteration.
                 # Let `prev_n_acceptable` and `prev_n_sample` be from the previous iteration.

                 # Need to initialize `prev_measure_grown` and `prev_measure_trimmed` before the loop.
                 # For i_exploration == 1, no previous growth happened. `prev_measure_grown` could be initialized to `measure_trimmed` (initial).

                 # Variables needed from previous iteration:
                 # `prev_n_acceptable`, `prev_n_sample`, `prev_measure_trimmed`, `prev_measure_grown` (measure before trim)

                 if i_exploration == 1:
                      prev_n_acceptable = 0
                      prev_n_sample = 0
                      prev_measure_trimmed = measure_trimmed # Initial measure
                      prev_measure_grown = measure_trimmed # Measure before growth in iter 1 is the initial measure

                 # Adaptive step (now placed after growth and sampling, using results from the *current* iteration)
                 # Let's move the adaptive step block.

                 # (Adaptive step will be moved below Sample step)

            # Where design variables aren't fixed, expand candidate solution box
            #   in both sides of each interval isotroply
            console.info('Growing candidate box... ');
            tic_grow = time.perf_counter()

            # Use the growth rate determined either initially or adaptively at the end of the previous loop.
            # So the adaptive step *should* happen at the end of the loop. Let's revert to matching the MATLAB placement.
            # This implies that the variables used in the adaptive step must somehow be available from the previous iteration
            # despite being calculated within the loop body of the current iteration in the code's structure.

            # Let's assume the adaptive block in the MATLAB code *is* using results from the previous iteration.
            # So, when i_exploration > 1, it uses `nAcceptable`, `nSample` from `i_exploration - 1`.
            # And `measureGrown` refers to the measure *before* trimming in `i_exploration - 1`.
            # And `measurePrevious` is the `measureTrimmed` from `i_exploration - 2`.

            # This is getting complex due to variable name ambiguity and placement.
            # Let's assume the simplest interpretation: The adaptive step at the start of iteration `k`
            # uses `purity` and `measure` values (`nAcceptable`/`nSample` and `measureTrimmed`) calculated at the end of iteration `k-1`.
            # And `measurePrevious` is `measureTrimmed` from iteration `k-2`. This requires tracking more variables.

            # Alternative simpler assumption: The adaptive step uses the results of the *current* iteration's sampling *after* the box is grown. This requires moving the adaptive step.
            # Let's try moving the adaptive step below the sampling/evaluation, as it makes more logical sense to adapt based on the current sample's purity.

            candidate_box_grown = design_box_grow_fixed(candidate_box, design_space_lower_bound, design_space_upper_bound, growth_rate)

            toc_grow = time.perf_counter()
            console.info('Elapsed time is %g seconds.\n', toc_grow - tic_grow)
            console.debug('- Current Growth Rate: %g\n', growth_rate)

            # Sample inside the current candidate box
            # get current number of samples
            n_sample = get_current_array_entry(options['NumberSamplesPerIterationExploration'], i_exploration)

            # Generate samples that are to be evaluated
            design_sample = sso_box_sub_generate_new_sample_points(
                candidate_box_grown,
                n_sample,
                options['SamplingMethodFunction'],
                options['SamplingMethodOptions'],
                console
            )

            # Evaluate the samples
            is_good_performance, is_physically_feasible, score, output_evaluation = sso_box_sub_evaluate_sample_points(
                design_evaluator,
                design_sample,
                console
            )

            # Label samples according to desired requirement spaces problem type
            is_acceptable, is_useful = sso_box_sub_label_samples_requirement_spaces(
                requirement_spaces_type,
                is_good_performance,
                is_physically_feasible,
                console
            )

            # Count number of labels
            n_acceptable, n_useful, n_acceptable_useful = sso_box_sub_count_label_acceptable_useful(
                is_acceptable,
                is_useful,
                console
            )

            # Compute candidate box measure *before* trimming (this is measureGrown)
            # Using the `candidate_box_grown` and the sampled points within it.
            # The measure function uses purity (nUseful/nSample). Here, nUseful and nSample are from the current sample.
            measure_grown = sso_box_sub_compute_candidate_box_measure(
                 candidate_box_grown,
                 n_sample,
                 n_useful, # Using n_useful from the current sample
                 options['MeasureFunction'],
                 options['MeasureOptions'],
                 console
            )


            # --- Adaptive step moved here ---
            if i_exploration > 1 and options['UseAdaptiveGrowthRate']:
                console.info('Adapting growth rate... ');
                tic_adapt = time.perf_counter()

                # Use the results of the *current* iteration's sampling (n_acceptable, n_sample)
                # and the measure change from the *previous* iteration (`measure_trimmed` at end of iter i-1 vs `measure_trimmed` at end of iter i-2)
                # No, the MATLAB code uses `measureGrown - measurePrevious` which is measure before trim (current) vs measure after trim (previous).
                # This still feels inconsistent with adaptive logic placement.

                # Let's strictly follow the MATLAB code's calculation:
                # `purity` is from the *current* sample (`n_acceptable`/`n_sample`).
                # `increaseMeasure` is the difference between `measure_grown` (current, before trim)
                # and `measurePrevious` (which was `measure_trimmed` at the end of the *previous* iteration).

                # Need to initialize `measurePrevious` before the loop (done) and update it at the end (will do).

                # Purity of the current sample
                purity = n_acceptable / n_sample if n_sample > 0 else 0
                purity = max(min(purity, options['MaximumGrowthPurity']), options['MinimumGrowthPurity'])

                increase_measure = measure_grown - measure_previous # measure before trim (current) - measure after trim (previous)

                # The MATLAB code has `increaseMeasureAcceptable = max(measureGrown*purity - measurePrevious,0);`
                # This seems to calculate the part of the measure increase attributable to acceptable points?
                # `fractionAcceptableIncreaseMeasure = increaseMeasureAcceptable/increaseMeasure;`
                # This is used as an input to the GrowthAdaptationFactorFunction.

                # Let's assume a simplified calculation for fractionAcceptableIncreaseMeasure if the original is unclear.
                # A plausible interpretation: purity * current grown measure represents a potential 'acceptable measure'.
                # Subtracting the previous trimmed measure gives an acceptable increase estimate.
                # The fraction is this acceptable increase relative to the total measure increase.

                # If `increase_measure` is zero or negative, the fraction is ill-defined or zero.
                increase_measure_acceptable = max(measure_grown * purity - measure_previous, 0)
                fraction_acceptable_increase_measure = increase_measure_acceptable / increase_measure if increase_measure > 0 else 0

                growth_adaptation_factor = options['GrowthAdaptationFactorFunction'](
                    purity,
                    options['TargetAcceptedRatioExploration'],
                    n_dimension,
                    fraction_acceptable_increase_measure,
                    **options['GrowthAdaptationFactorOptions']
                )
                growth_adaptation_factor = max(min(growth_adaptation_factor, options['MaximumGrowthAdaptationFactor']), options['MinimumGrowthAdaptationFactor'])

                growth_rate = growth_adaptation_factor * growth_rate
                growth_rate = max(min(growth_rate, options['MaximumGrowthRate']), options['MinimumGrowthRate'])

                toc_adapt = time.perf_counter()
                console.info('Elapsed time is %g seconds.\n', toc_adapt - tic_adapt)
                # --- End of Adaptive step ---


            # Box may have grown too much + "unlucky sampling" w/ no good
            # points, go back in this case
            if n_acceptable == 0 or n_useful == 0 or n_acceptable_useful == 0:
                console.warning('SSOOptBox:BadSampling No good/useful points found, rolling back and reducing growth rate to minimum...')

                # In MATLAB, they log and then `continue` to the next iteration,
                # effectively skipping the trimming and using the `candidateBox` from the start of this iteration
                # (which was the `candidateBoxTrimmed` from the previous iteration).
                # They also set `growthRate` to minimum, but the adaptive step already handles rate reduction.
                # Setting `options.MaximumGrowthRate = growthRate;` in MATLAB seems to hard-limit the rate permanently.
                # Let's just revert the candidate box and potentially adjust the growth rate downwards if not already minimal.

                # Revert candidate box to the state before growth in this iteration
                # The box at the start of this iteration `i_exploration` was `candidate_box` (which was `candidateBoxTrimmed` from i_exploration - 1).
                candidate_box_trimmed = candidate_box.copy() # Effectively no trimming happened
                measure_trimmed = measure_previous # The measure remains the same as before the failed growth/sample step

                # Ensure growth rate is low for the next attempt
                growth_rate = options['MinimumGrowthRate']
                # Optionally, also set the maximum allowed rate for future iterations if this failure is severe
                # options['MaximumGrowthRate'] = growth_rate # This matches the MATLAB code's apparent intent

                if is_output_iteration_data:
                    console.info('Logging relevant information... ');
                    tic_log = time.perf_counter()

                    # Log data for the failed iteration attempt
                    iteration_data.append({
                        'EvaluatedDesignSamples': design_sample,
                        'EvaluationOutput': output_evaluation,
                        'Phase': 1,
                        'GrowthRate': growth_rate, # Log the rate used in this failed iteration
                        'DesignScore': score,
                        'IsGoodPerformance': is_good_performance,
                        'IsPhysicallyFeasible': is_physically_feasible,
                        'IsAcceptable': is_acceptable,
                        'IsUseful': is_useful,
                        'CandidateBoxBeforeTrim': candidate_box_grown.copy(),
                        'CandidateBoxAfterTrim': candidate_box_trimmed.copy(), # Log the reverted box
                    })
                    log_index += 1

                    toc_log = time.perf_counter()
                    console.info('Elapsed time is %g seconds.\n', toc_log - tic_log)

                # Prepare for next iteration - candidate_box is already the reverted one.
                i_exploration += 1
                measure_previous = measure_trimmed # Update previous measure for the next iteration's check
                continue # Skip trimming and convergence check, move to the next iteration

            # If sampling was successful (found good/useful points)
            # Modification Step A - Trimming: Remove Bad Points
            # find trimming order
            # The MATLAB code calculates `orderTrim` using `~isAcceptable`.
            # It then passes this order to `sso_box_sub_trimming_operation` which uses `labelViable = isAcceptable & isUseful`.
            # This suggests the order is based on *unacceptable* points, but trimming targets making the box contain *viable* points.
            # The `trimmingMethodFunction` needs to correctly interpret the `orderTrim` and `labelViable`.
            # Let's assume `orderTrim` specifies the points that are candidates for removal, and the trimming function
            # uses `labelViable` to decide which points *must* be kept.
            order_trim = options['TrimmingOrderFunction'](~is_acceptable, score, **options['TrimmingOrderOptions'])

            candidate_box_trimmed, measure_trimmed = sso_box_sub_trimming_operation(
                candidate_box_grown, # Trim from the grown box
                measure_grown,       # Measure of the grown box
                design_sample,
                is_acceptable,
                is_useful,
                order_trim,
                options['TrimmingOperationFunction'],
                trimming_operation_options,
                console
            )

            # Apply leanness trimming if specified for each trim
            if apply_leanness_each_trim:
                 console.info('Applying leanness trimming (each trim)... ')
                 tic_leanness = time.perf_counter()
                 # Leanness trimming order is typically based on 'useless' points by score.
                 # Assuming trimming_order handles the 'OrderPreference' argument.
                 trimming_order_leanness = trimming_order(~is_useful, score, OrderPreference='score-low-to-high')
                 candidate_box_trimmed = box_trimming_leanness(
                     design_sample, is_useful, trimming_order_leanness, candidate_box_trimmed)

                 # Re-calculate measure after leanness trimming
                 # Need to check which of the *original* samples are still inside the *new* trimmed box.
                 inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
                 n_useful_inside_new = np.sum(inside_box_new.flatten() & is_useful) # Use .flatten() for boolean indexing
                 n_inside_new = np.sum(inside_box_new)

                 purity_after_leanness = n_useful_inside_new / n_inside_new if n_inside_new > 0 else 0

                 measure_trimmed = sso_box_sub_compute_candidate_box_measure(
                     candidate_box_trimmed,
                     n_inside_new, # Number of samples remaining inside the box
                     n_useful_inside_new,
                     options['MeasureFunction'],
                     options['MeasureOptions'],
                     console
                 )
                 toc_leanness = time.perf_counter()
                 console.info('Elapsed time is %g seconds.\n', toc_leanness - tic_leanness)


            # Convergence Criteria for Phase I
            console.info('Checking convergence... ');
            tic_converge = time.perf_counter()

            # Stop phase I if measure doesn't change significantly from step to step
            # Comparison is between `measure_trimmed` from *this* iteration and `measure_previous`
            # which was `measure_trimmed` from the *previous* iteration.
            measure_change_relative = abs(measure_trimmed - measure_previous) / (measure_trimmed if measure_trimmed != 0 else 1) # Avoid division by zero

            if i_exploration >= options['MaxIterExploration']:
                has_converged_exploration = True
                console.info('Phase I converged due to maximum number of iterations reached.')
            elif (not options['FixIterNumberExploration'] and
                  measure_change_relative < options['ToleranceMeasureChangeExploration']):
                has_converged_exploration = True
                console.info('Phase I converged due to small measure change.')

            toc_converge = time.perf_counter()
            console.info('Elapsed time is %g seconds.\n', toc_converge - tic_converge)

            if is_output_iteration_data:
                # Save Data for this iteration
                console.info('Logging relevant information... ');
                tic_log = time.perf_counter()

                iteration_data.append({
                    'EvaluatedDesignSamples': design_sample.copy(),
                    'EvaluationOutput': output_evaluation, # Assume output_evaluation is copyable or immutable
                    'Phase': 1,
                    'GrowthRate': growth_rate,
                    'DesignScore': score.copy(),
                    'IsGoodPerformance': is_good_performance.copy(),
                    'IsPhysicallyFeasible': is_physically_feasible.copy(),
                    'IsAcceptable': is_acceptable.copy(),
                    'IsUseful': is_useful.copy(),
                    'CandidateBoxBeforeTrim': candidate_box_grown.copy(),
                    'CandidateBoxAfterTrim': candidate_box_trimmed.copy(),
                })
                log_index += 1

                toc_log = time.perf_counter()
                console.info('Elapsed time is %g seconds.\n', toc_log - tic_log)

            # Prepare for next iteration
            console.info('Done with iteration %d!\n', i_exploration)

            candidate_box = candidate_box_trimmed.copy() # Update candidate_box for the next iteration's growth
            measure_previous = measure_trimmed # Update for the next iteration's convergence check
            i_exploration += 1

    console.info('\nDone with Phase I - Exploration in iteration %d!\n\n', i_exploration - 1)


    # Phase II - Consolidation
    console.info('=' * 120)
    console.info('Initiating Phase II - Consolidation...\n')

    # Iteration start
    convergence_consolidation = False
    if options['FixIterNumberConsolidation'] and (options['MaxIterConsolidation'] == 0):
        convergence_consolidation = True
        console.info('Phase II skipped due to MaxIterConsolidation = 0 and FixIterNumberConsolidation = True.')


    i_consolidation = 1
    tolerance_purity_consolidation = options['TolerancePurityConsolidation'] # Get tolerance for clarity

    while ((not convergence_consolidation) and (i_consolidation <= options['MaxIterConsolidation'])):
        console.info('=' * 120)
        console.info('Initiating Phase II - Consolidation: Iteration %d...\n', i_consolidation)

        # get current number of samples
        n_sample = get_current_array_entry(options['NumberSamplesPerIterationConsolidation'], i_consolidation)

        # Generate samples that are to be evaluated (sample inside the *current* candidate box)
        design_sample = sso_box_sub_generate_new_sample_points(
            candidate_box, # Sample inside the box from the end of the previous phase/iteration
            n_sample,
            options['SamplingMethodFunction'],
            options['SamplingMethodOptions'],
            console
        )

        # Evaluate the samples
        is_good_performance, is_physically_feasible, score, output_evaluation = sso_box_sub_evaluate_sample_points(
            design_evaluator,
            design_sample,
            console
        )

        # Label samples according to desired requirement spaces problem type
        is_acceptable, is_useful = sso_box_sub_label_samples_requirement_spaces(
            options['RequirementSpacesType'],
            is_good_performance,
            is_physically_feasible,
            console
        )

        # Count number of labels
        n_acceptable, n_useful, n_acceptable_useful = sso_box_sub_count_label_acceptable_useful(
            is_acceptable,
            is_useful,
            console
        )

        # No viable design found; throw error (or warning and handle)
        # In Phase II, failure to find viable points within the current box is critical.
        if n_acceptable == 0 or n_useful == 0 or n_acceptable_useful == 0:
            console.critical('SSOOptBox:BadSampling No good/useful points found inside the current candidate box. Please retry process with different parameters / looser requirements.')
            # Depending on desired behavior, could raise an exception here.
            # For now, we log critical and the loop will terminate if MaxIterConsolidation is reached.
            # However, the subsequent trimming step might fail if there are no acceptable points.
            # Let's handle this before trimming. If no acceptable points, no trimming can improve purity.
            # The MATLAB code proceeds to trimming even if nAcceptable is 0, which might cause issues in trimming_method_function.
            # Let's add a check here. If no acceptable points, we might as well stop or at least skip trimming.

            # Let's match the MATLAB flow - proceed to trimming. The trimming function needs to handle the case of no viable points.
            # The mock `sso_box_sub_trimming_operation` does have basic handling for `np.sum(label_viable) == 0`.

        # Compute candidate box measure (of the *current* candidate box, using current sample purity)
        measure = sso_box_sub_compute_candidate_box_measure(
            candidate_box,
            n_sample,
            n_useful, # Using n_useful from the current sample
            options['MeasureFunction'],
            options['MeasureOptions'],
            console
        )

        # Convergence Check - Purity
        current_purity = n_acceptable / n_sample if n_sample > 0 else 0
        if (not options['FixIterNumberConsolidation'] and
            current_purity >= tolerance_purity_consolidation):
            convergence_consolidation = True
            console.info('Phase II converged due to purity tolerance reached.')

        # Modification Step A (Trimming): Remove Bad Points
        # Only trim if not yet converged by purity (unless FixIterNumberConsolidation is true, then trim every iter)
        # The MATLAB code trims if `~convergenceConsolidation` is true. This means trimming stops *as soon as* purity tolerance is met, even if MaxIterConsolidation is not reached yet and FixIterNumberConsolidation is false. If FixIterNumberConsolidation is true, trimming happens in every iteration up to MaxIterConsolidation.
        # This logic is slightly complex. Let's stick to: trim unless purity convergence is met AND FixIterNumberConsolidation is false.
        perform_trimming_consolidation = not (convergence_consolidation and not options['FixIterNumberConsolidation'])

        candidate_box_trimmed = candidate_box.copy() # Initialize to current box
        measure_trimmed = measure # Initialize to current measure

        if perform_trimming_consolidation:
            order_trim = options['TrimmingOrderFunction'](~is_acceptable, score, **options['TrimmingOrderOptions'])
            candidate_box_trimmed, measure_trimmed = sso_box_sub_trimming_operation(
                candidate_box, # Trim from the candidate box at start of iter
                measure,     # Measure of the candidate box at start of iter
                design_sample,
                is_acceptable,
                is_useful,
                order_trim,
                options['TrimmingOperationFunction'],
                trimming_operation_options,
                console
            )

            # Apply leanness trimming if specified for each trim in Phase II
            if apply_leanness_each_trim:
                 console.info('Applying leanness trimming (each trim) in Phase II... ')
                 tic_leanness = time.perf_counter()
                 trimming_order_leanness = trimming_order(~is_useful, score, OrderPreference='score-low-to-high')
                 candidate_box_trimmed = box_trimming_leanness(
                     design_sample, is_useful, trimming_order_leanness, candidate_box_trimmed)

                 # Re-calculate measure after leanness trimming
                 inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
                 n_useful_inside_new = np.sum(inside_box_new.flatten() & is_useful)
                 n_inside_new = np.sum(inside_box_new)
                 purity_after_leanness = n_useful_inside_new / n_inside_new if n_inside_new > 0 else 0
                 measure_trimmed = sso_box_sub_compute_candidate_box_measure(
                     candidate_box_trimmed,
                     n_inside_new,
                     n_useful_inside_new,
                     options['MeasureFunction'],
                     options['MeasureOptions'],
                     console
                 )
                 toc_leanness = time.perf_counter()
                 console.info('Elapsed time is %g seconds.\n', toc_leanness - tic_leanness)


        # Convergence check - Number of Iterations
        if i_consolidation >= options['MaxIterConsolidation']:
            convergence_consolidation = True
            console.info('Phase II converged due to maximum number of iterations reached.')


        if is_output_iteration_data:
            # Save Data for this iteration
            console.info('Logging relevant information... ');
            tic_log = time.perf_counter()

            iteration_data.append({
                'EvaluatedDesignSamples': design_sample.copy(),
                'EvaluationOutput': output_evaluation,
                'Phase': 2,
                'GrowthRate': None, # No growth rate in Phase 2
                'DesignScore': score.copy(),
                'IsGoodPerformance': is_good_performance.copy(),
                'IsPhysicallyFeasible': is_physically_feasible.copy(),
                'IsAcceptable': is_acceptable.copy(),
                'IsUseful': is_useful.copy(),
                'CandidateBoxBeforeTrim': candidate_box.copy(), # Box at start of iteration
                'CandidateBoxAfterTrim': candidate_box_trimmed.copy(), # Box after trimming (or original if no trim)
            })
            log_index += 1

            toc_log = time.perf_counter()
            console.info('Elapsed time is %g seconds.\n', toc_log - tic_log)

        # Prepare for next iteration
        console.info('Done with iteration %d!\n', i_consolidation)
        candidate_box = candidate_box_trimmed.copy() # Update candidate box for the next iteration
        i_consolidation += 1

    console.info('\nDone with Phase II - Consolidation in iteration %d!\n\n', i_consolidation - 1)


    # Check for the leanness condition (final trim)
    if apply_leanness_final_trim:
        console.info('Applying final leanness trimming... ')
        tic_leanness_final = time.perf_counter()

        # Leanness trimming needs the last set of samples and their labels/scores.
        # These are available from the last iteration of Phase II.
        # If Phase II was skipped, need to use the last data from Phase I.
        # Check if iteration_data is not empty.
        if log_index > 0:
             last_iter_data = iteration_data[-1]
             last_design_sample = last_iter_data['EvaluatedDesignSamples']
             last_is_useful = last_iter_data['IsUseful']
             last_score = last_iter_data['DesignScore']

             trimming_order_leanness_final = trimming_order(~last_is_useful, last_score, OrderPreference='score-low-to-high')
             candidate_box = box_trimming_leanness(
                 last_design_sample, last_is_useful, trimming_order_leanness_final, candidate_box)

        else:
             console.warning("No iteration data available for final leanness trim.")


        toc_leanness_final = time.perf_counter()
        console.info('Elapsed time is %g seconds.\n', toc_leanness_final - tic_leanness_final)


    # Return outputs
    # MATLAB returns problemData and iterationData only if nargout is >= 2 or >= 3.
    # In Python, we can use inspection (like `len(sys.argv)`) but it's better to always return them
    # and let the caller decide what to use, or add output flags to the function signature.
    # Let's always return all three as per the Python function signature defined.

    return candidate_box, problem_data, iteration_data


# Example usage (requires implementing the placeholder dependencies)
if __name__ == '__main__':
    # Mock implementation details for placeholders to make the example runnable
    class MockDesignEvaluator(DesignEvaluatorBase):
        def evaluate(self, design_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
            n_sample, n_dim = design_samples.shape
            # Example criteria: points are 'good' if sum of dimensions < threshold
            # and 'feasible' if within a certain sub-box.
            threshold = 0.5 * n_dim
            performance_deficit = np.sum(design_samples, axis=1, keepdims=True) - threshold
            # Feasibility within a sub-box [0.2, 0.8] in each dimension
            feasibility_lower = np.array([0.2] * n_dim)
            feasibility_upper = np.array([0.8] * n_dim)
            # Deficit is max(0, lower_violation, upper_violation)
            lower_violation = np.max(feasibility_lower - design_samples, axis=1, keepdims=True)
            upper_violation = np.max(design_samples - feasibility_upper, axis=1, keepdims=True)
            physical_feasibility_deficit = np.maximum(lower_violation, upper_violation)

            evaluation_output = {'sum_dims': np.sum(design_samples, axis=1)}
            return performance_deficit, physical_feasibility_deficit, evaluation_output

    # Simple mock for trimming_method_function that just calculates bounds of viable points
    def mock_trimming_operation(
        design_sample,
        label_viable,
        order_trim,
        candidate_box,
        options
    ) -> Tuple[np.ndarray, float]:
        """Simple mock trimming: set box bounds to min/max of viable points."""
        console = ConsoleLogging('DEBUG') # Use a local console for logging within mock

        if np.sum(label_viable) > 0:
            viable_points = design_sample[label_viable, :]
            trimmed_box_min = np.min(viable_points, axis=0)
            trimmed_box_max = np.max(viable_points, axis=0)
            candidate_box_trimmed = np.vstack((trimmed_box_min, trimmed_box_max))

            # Recalculate measure based on the new box and purity *within the new box*
            inside_box_new = is_in_design_box(design_sample, candidate_box_trimmed)
            n_useful_inside_trimmed = np.sum(inside_box_new.flatten() & (label_viable.flatten())) # Use viable as useful here for simplicity
            n_inside_trimmed = np.sum(inside_box_new)
            purity_inside_trimmed = n_useful_inside_trimmed / n_inside_trimmed if n_inside_trimmed > 0 else 0

            measure_function = options.get('MeasureFunction', lambda box, purity, *a: np.prod(box[1, :] - box[0, :]) * purity)
            measure_options = options.get('MeasureOptions', {})

            measure_trimmed = measure_function(candidate_box_trimmed, purity_inside_trimmed, **measure_options)

        else:
            # If no viable points, return a zero-volume box at the center of the previous box
            center = np.mean(candidate_box, axis=0)
            candidate_box_trimmed = np.vstack((center, center))
            measure_trimmed = 0.0 # Zero measure

        # In a real trimming operation, you'd iterate through `order_trim` and
        # decide whether removing a point allows shrinking the box while keeping viable points.
        # This mock doesn't do that, it's a simpler "fit to viable points" logic.
        console.debug("Mock trimming operation resulted in box: %s", candidate_box_trimmed)
        return candidate_box_trimmed, measure_trimmed


    # Override some placeholder functions for the example
    # Make the mock trimming operation use our simple mock
    def sso_stochastic_options_example(opt_type: str) -> Dict[str, Any]:
        options = sso_stochastic_options(opt_type)
        options['TrimmingOperationFunction'] = mock_trimming_operation
        # Increase iterations for a more visible exploration phase
        options['MaxIterExploration'] = 20
        options['MaxIterConsolidation'] = 30
        options['NumberSamplesPerIterationExploration'] = 200
        options['NumberSamplesPerIterationConsolidation'] = 500
        options['LoggingLevel'] = 'DEBUG' # More verbose logging for example
        return options

    # Replace the default sso_stochastic_options with our example version for this run
    original_sso_stochastic_options = sso_stochastic_options
    sso_stochastic_options = sso_stochastic_options_example

    # Define problem inputs
    n_design_variable = 2
    design_evaluator = MockDesignEvaluator()
    initial_box = np.array([[0.4, 0.4], [0.6, 0.6]]) # Start with a small box
    design_space_lower_bound = np.array([[0.0, 0.0]])
    design_space_upper_bound = np.array([[1.0, 1.0]])

    # Run the optimization
    print("Starting SSO optimization...")
    try:
        optimal_box, problem_data_out, iteration_data_out = sso_box_stochastic(
            design_evaluator,
            initial_box,
            design_space_lower_bound,
            design_space_upper_bound,
            # Pass options as keyword arguments
            # RequirementSpacesType='feasible',
            # GrowthRate=0.05,
            # UseAdaptiveGrowthRate=False
        )

        print("\nOptimization finished.")
        print("\nOptimal Candidate Box:")
        print(optimal_box)

        # print("\nProblem Data:")
        # print(problem_data_out)

        # print("\nIteration Data (first 2 iterations):")
        # for i, data in enumerate(iteration_data_out[:2]):
        #     print(f"--- Iteration {i+1} ---")
        #     for key, value in data.items():
        #         if isinstance(value, np.ndarray):
        #             print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        #         else:
        #             print(f"{key}: {type(value)}")
        #     print("-" * 10)

    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")

    finally:
        # Restore original sso_stochastic_options after example run
        sso_stochastic_options = original_sso_stochastic_options
