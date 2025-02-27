import numpy as np
from numpy._typing import NDArray
from scipy import optimize


def find_threshold(
    vgs: NDArray[np.float64],
    ids: NDArray[np.float64],
    vdd: float,
    threshold_current_ratio = 1000
) -> float:
    index = np.searchsorted(vgs, vdd)
    ids_high = ids[index]
    threshold_ids = ids_high / threshold_current_ratio
    threshold_index = np.searchsorted(ids, threshold_ids)
    threshold_voltage = float(vgs[threshold_index])
    return threshold_voltage

def find_threshold2(
    vgs: NDArray[np.float64],
    ids: NDArray[np.float64],
    vdd: float,
    threshold_current_ratio = 1000,
    bound_safety = 0
) -> float:
    initial_guess = find_threshold(vgs, ids, vdd, threshold_current_ratio)
    saturation_threshold = vdd - initial_guess
    sat_low_index = np.searchsorted(vgs, initial_guess)
    sat_high_index = np.searchsorted(vgs, saturation_threshold) + 1
    sat_width = sat_high_index - sat_low_index
    bound_increment = int(sat_width * bound_safety / 2)
    bound_sat_low = sat_low_index + bound_increment
    bound_sat_high = sat_high_index + bound_increment
    # sat_range = range(sat_low_index, sat_high_index + 1)
    bound_sat_range = np.arange(bound_sat_low, bound_sat_high)
    ids_sat = ids[bound_sat_range]
    vgs_sat = vgs[bound_sat_range]
    quad_model = np.polyfit(vgs_sat, ids_sat, 2)
    a, b, c = quad_model[0], quad_model[1], quad_model[2]
    vertex = -b / (2 * a)
    return vertex
