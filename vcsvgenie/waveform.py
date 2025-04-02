from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame


@dataclass
class WaveForm:
    """
    Stores 2D datapoints. Labeled too.
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    title: str


def construct_waveforms(
        waveform_dataframe: DataFrame, titles: List[str]
) -> List[WaveForm]:
    """
    Constructs a list of read_waveforms from a read_dataframe and list of read_titles.
    :param waveform_dataframe: Dataframe of read_waveforms
    :param titles: List of read_titles
    :return: List of read_waveforms taken from the read_dataframe, titled as they are in the read_titles list
    """
    waveforms: List[WaveForm] = list()
    for idx, title in enumerate(titles):
        x, y = waveform_dataframe.iloc[:, 2 * idx].to_numpy(
            dtype=np.float64
        ), waveform_dataframe.iloc[:, 2 * idx + 1].to_numpy(dtype=np.float64)
        waveform = WaveForm(x, y, title)
        waveforms.append(waveform)
    return waveforms


def linear_interpolation(
        x_initial: float,
        x_final: float,
        y_initial: float,
        y_final: float,
        threshold: float,
        threshold_axis: Literal['x', 'y']
) -> float:
    """
    Linear interpolation between two points.
    :param x_initial: Initial x value
    :param x_final: Final x value
    :param y_initial: Initial y value
    :param y_final: Final y value
    :param threshold: Threshold value on one axis to find the value on the other axis for
    :param threshold_axis: Threshold value axis (either x or y)
    """
    if threshold_axis == 'x':
        primary_initial = x_initial
        primary_final = x_final
        secondary_initial = y_initial
        secondary_final = y_final
    else:
        primary_initial = y_initial
        primary_final = y_final
        secondary_initial = x_initial
        secondary_final = x_final

    if not (min(primary_initial, primary_final) <= threshold <= max(primary_initial, primary_final)):
        raise ArithmeticError(
            f"Extrapolative threshold: Initial = {primary_initial}, Final = {primary_final}, Threshold = {threshold}"
        )

    proportion = (threshold - primary_initial) / (primary_final - primary_initial)
    interpolated_secondary = proportion * (secondary_final - secondary_initial) + secondary_initial
    return interpolated_secondary

