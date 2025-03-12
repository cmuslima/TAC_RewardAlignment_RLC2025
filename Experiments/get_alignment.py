from scipy.stats import kendalltau 
import numpy as np
def check_existence_of_any_ties(data):
    """
    Checks if there are any ties (all elements have the same value) in a dataset.
    Args:
        data (numpy.ndarray): The dataset to check for ties.
    Returns:
        bool: True if there are ties, False otherwise.
    """
    min = np.min(data)
    max = np.max(data)
    if min == max:
        return True
    else:
        return False

def handle_nans_in_kendall_tau_calculation(statistic, human_rf_data, fitness_data):
    """
    Handles Not-a-Number (NaN) values encountered during Kendall's Tau calculation for ordinal data.

    This function checks for NaN values in the calculated Kendall's Tau statistic. If a NaN is found,
    it assumes there might be issues with the data. It then calls an external function `check_existence_of_any_ties`
    to determine if both human and fitness reward data have only ties (identical values).

    * If there are ties in both human and fitness data, the function assumes perfect agreement (Kendall's Tau of 1).
    * Otherwise, it assumes no agreement (Kendall's Tau of 0).

    For valid Kendall's Tau statistics (not NaN), the function directly returns the value.

    Args:
        statistic: The calculated Kendall's Tau statistic (might be NaN).
        human_rf_data: Human reward function data for the current run.
        fitness_data: Fitness reward function data for the current run.

    Returns:
        float: The Kendall's Tau value, with substitutions for NaN values based on tie information.
    """
    if str(statistic) == 'nan':
        ties_rf = check_existence_of_any_ties(human_rf_data)
        ties_fitness = check_existence_of_any_ties(fitness_data)
        if ties_rf and ties_fitness:
            return 1
        else:
            return 0      
    else:
        return statistic


def get_trajectory_alignment_coefficient_score(human_rf_data, fitness_data):
    statistic, _ = kendalltau(human_rf_data, fitness_data) 

    statistic = handle_nans_in_kendall_tau_calculation(statistic, human_rf_data, fitness_data)
    return statistic

