from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from vcsvgenie.waveform import WaveForm


class DCResult:
    """
    Harness class for future DC Analysis attributes and methods.
    Right now, this doesn't do much. There's a subclass for SRAM read analysis.
    """
    def __init__(
            self,
            signals: List[WaveForm],
            name: str = "DC Results",
    ):
        """
        Iniitalizes the DCResults object.
        The signals passed to this method will be processed (the waveform points will be sorted by x series)
        :param signals: List of read_waveforms (ostensibly defined from DCResultSpecification)
        :param name: Name of the read_result
        """
        self.signals: Dict[str, WaveForm] = dict()
        for signal in signals:
            x = signal.x
            y = signal.y
            sorted_x, x_idxes = np.sort(x), np.argsort(x)
            sorted_y = y[x_idxes]
            self.signals[signal.title] = WaveForm(sorted_x, sorted_y, signal.title)

class DCResultSpecification:
    """
    Specification class for DCResults.
    May be extended in future versions.
    """
    def __init__(
            self,
            signals: List[str]
    ):
        """
        Initializes the DCResultSpecification object.
        :param signals: List of waveform read_titles to include in spec
        """
        self.signals = signals

    def interpret(self, waveforms: List[WaveForm], name: str = "DC Results") -> DCResult:
        """
        Interprets a list of read_waveforms into a DCResult object.
        :param waveforms: List of read_waveforms to select from. (Might be a superset of intended selection)
        :param name: Name of the DCResult
        :return: A DCResult containing the read_waveforms as specified in this object, named per the argument
        """
        result_waveforms: List[WaveForm] = list()
        for waveform in waveforms:
            if waveform.title in self.signals:
                result_waveforms.append(waveform)

        return DCResult(result_waveforms, name)

def reconcile(w1: WaveForm, w2: WaveForm) -> Tuple[WaveForm, WaveForm]:
    """
    Augments a pair of read_waveforms such that their x axes are identical, allowing direct comparison of points.
    :param w1: First waveform
    :param w2: Second waveform
    """
    y1_p = np.interp(w2.x, w1.x, w1.y)
    y2_p = np.interp(w1.x, w2.x, w2.y)
    x1 = np.append(w1.x, w2.x)
    x2 = np.append(w2.x, w1.x)
    y1 = np.append(w1.y, y1_p)
    y2 = np.append(w2.y, y2_p)
    sorted_x1, sort_index_1 = np.sort(x1), np.argsort(x1)
    sorted_x2, sort_index_2 = np.sort(x2), np.argsort(x2)
    sorted_y1 = y1[sort_index_1]
    sorted_y2 = y2[sort_index_2]
    return WaveForm(sorted_x1, sorted_y1, w1.title), WaveForm(sorted_x2, sorted_y2, w2.title)

def argfind_intercept(w1: WaveForm, w2: WaveForm) -> int:
    """
    Finds the index corresponding to where the intercept point between two read_waveforms should be inserted.
    Returns -1 if the transition point is not found.
    (This method does not attempt extrapolation beyond dataset in positive direction.)
    :param w1: First waveform
    :param w2: Second waveform
    """
    w1_is_upper = w1.y >= w2.y
    dw1_is_upper = np.bitwise_xor(w1_is_upper[:-1], w1_is_upper[1:])
    try:
        transition = int(np.argwhere(dw1_is_upper)[0] + 1)
        return transition
    except IndexError:
        return -1
    except Exception as e:
        print("Unknown error")
        print(e)
        return -1

def argfind_eye(w1: WaveForm, w2: WaveForm) -> Tuple[int, int]:
    """
    Finds the "eye" in which a Noise Margin measurement (e.g., for SRAM) should be taken.
    :param w1: First waveform
    :param w2: Second waveform
    :return: The two indices corresponding to the beginning (inclusive) and the end (exclusive) of the eye
    """
    w1_is_upper = w1.y >= w2.y
    dw1_is_upper = np.bitwise_xor(w1_is_upper[:-1], w1_is_upper[1:])
    eye_transition = np.argwhere(dw1_is_upper) + 1
    half_len = w1.y.shape[0] / 2
    dist = np.abs(eye_transition - half_len)
    closest_transition = np.argmin(dist)
    if closest_transition == 0:
        return 0, int(eye_transition[closest_transition])
    else:
        return int(eye_transition[closest_transition - 1]), int(eye_transition[closest_transition])

def unit_line_through_y_intercept(w: WaveForm, idx: int | NDArray[np.int32]) -> np.float64 | NDArray[np.float64]:
    """
    Calculates b for line(s) y = x + b intercepting the waveform w at indexes idx.
    This is useful for noise margin calculation.
    :param w: Waveform to reference
    :param idx: Index(es) of the datapoint(s) for which to find intercepting lines
    :return: The y-intercept(s) of the unit-slope lines defined by the waveform points (b in y = x + b)
    """
    x = w.x[idx]
    y = w.y[idx]
    return y - x


def cast_onto_unit_line(x: NDArray[np.float64], b: NDArray[np.float64] | np.float64) -> NDArray[np.float64]:
    """
    Projects datapoints x onto a unit line y = x + b
    This is useful for noise margin calculation.
    :param x: Data points to project
    :param b: Defines the unit-slope line on which to project the datapoints
    :return: Projected data
    """
    if np.isscalar(b):
        return x + b

    y = np.zeros((b.shape[0], x.shape[0]), dtype=np.float64)
    for idx in range(b.shape[0]):
        y[idx, :] = x + b[idx]

    return y

def linear_intercept(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculates the point intercept between two line segments.
    This is useful for noise margin calculation.
    These line segments should be intersecting already (this code does not check).
    :param p1: Start point of first line segment
    :param p2: End point of first line segment
    :param p3: Start point of second line segment
    :param p4: End point of second line segment
    :return: Intercept between two line segments (x, y) as floats
    """
    l1m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    l1b = p1[1] - l1m * p1[0]
    l2m = (p4[1] - p3[1]) / (p4[0] - p3[0])
    l2b = p3[1] - l2m * p3[0]
    intercept_x = (l1b - l2b) / (l2m - l1m)
    intercept_y = l1m * intercept_x + l1b
    return intercept_x, intercept_y


class ReadSRAMNoiseMarginResult(DCResult):
    """
    Class for calculating the Read SRAM Noise Margin.
    May be extended/split in future versions to cover Hold Noise margin (if
    changes needed)
    """
    def __init__(self, signals: List[WaveForm], name: str = "RSNM Results"):
        """
        Initializes the ReadSRAMNoiseMarginResult.
        :param signals: List of WaveForm objects (ostensibly from spec)
        :param name: Name of the read_result
        """
        if len(signals) != 2:
            raise Exception("This RSNM ResultSpecification class only supports analyzing a pair of signals")
        super().__init__(signals, name)

        self.signal1 = self.signals[signals[0].title]
        self.signal2 = self.signals[signals[1].title]
        self.square_dim: float = 0
        self.square_dim_anchor: int = -1

    def truncate(self):
        """
        Truncates the underlying read_waveforms to remove datapoints that both read_waveforms clearly would not share.
        (i.e., if one waveform's x-series includes 0.01, 0.02, 0.03, 0.04, 0.05 and the other includes 0.05, ... then
        the first four points of the first waveform are removed. A similar process happens to the end of the read_waveforms
        as well.)

        This function should only be run once. Multiple runs should not have any effect.
        """
        if self.signal1.x[0] > self.signal2.x[0]:
            start1 = self.signal1.x[0]
            argstart2 = np.searchsorted(self.signal2.x, start1)
            argstart1 = 0
        else:
            start2 = self.signal2.x[0]
            argstart1 = np.searchsorted(self.signal1.x, start2)
            argstart2 = 0

        if self.signal1.x[-1] > self.signal2.x[-1]:
            end2 = self.signal2.x[-1]
            argend1 = min(int(np.searchsorted(self.signal1.x, end2)), self.signal1.x.shape[0] - 1)
            argend2 = self.signal2.x.shape[0] - 1
        else:
            end1 = self.signal1.x[-1]
            argend2 = min(int(np.searchsorted(self.signal2.x, end1)), self.signal2.x.shape[0] - 1)
            argend1 = self.signal1.x.shape[0] - 1

        truncated_signal1_x = self.signal1.x[argstart1:argend1+1]
        truncated_signal1_y = self.signal1.y[argstart1:argend1+1]
        truncated_signal2_x = self.signal2.x[argstart2:argend2+1]
        truncated_signal2_y = self.signal2.y[argstart2:argend2+1]
        self.signal1 = WaveForm(
            truncated_signal1_x,
            truncated_signal1_y,
            self.signal1.title,
        )
        self.signal2 = WaveForm(
            truncated_signal2_x,
            truncated_signal2_y,
            self.signal2.title,
        )

    def reconcile(self):
        """
        Reconciles the two underlying read_waveforms so that their x axes include the same points, allowing direct
        point comparisons.

        Calls the reconcile method from outside the class to do this.
        """
        self.signal1, self.signal2 = reconcile(self.signal1, self.signal2)

    def calculate_square_dim(self) -> Tuple[float, int]:
        """
        Calculates the noise margin. This is done by finding the size of the largest square that one can inscribe
        within the eye defined by the two read_waveforms within this results object.
        """
        sidx, fidx = argfind_eye(self.signal1, self.signal2)
        y_intercepts = unit_line_through_y_intercept(self.signal1, np.arange(sidx, fidx))
        unit_line_y_value_array = cast_onto_unit_line(self.signal1.x, y_intercepts)
        square_dim = np.zeros(fidx, dtype=np.float64)
        for idx in range(sidx, fidx):
            unit_line_y_values = unit_line_y_value_array[idx - sidx, :]
            unit_line = WaveForm(self.signal1.x, unit_line_y_values, "Unit Line")
            unit_line_intercept_idx = argfind_intercept(self.signal2, unit_line)
            xs = float(self.signal2.x[unit_line_intercept_idx - 1])
            xf = float(self.signal2.x[unit_line_intercept_idx])
            ys = float(self.signal2.y[unit_line_intercept_idx - 1])
            yf = float(self.signal2.y[unit_line_intercept_idx])
            us = float(xs + y_intercepts[idx - sidx])
            uf = float(xf + y_intercepts[idx - sidx])

            p1 = (xs, ys)
            p2 = (xf, yf)
            p3 = (xs, us)
            p4 = (xf, uf)

            sx, sy = linear_intercept(p1, p2, p3, p4)
            dim = sx - self.signal1.x[idx]

            def debug_plot():
                plt.figure()
                plt.plot(self.signal1.x, self.signal1.y)
                plt.plot(self.signal2.x, self.signal2.y)
                plt.plot(self.signal1.x, unit_line_y_values)
                plt.scatter((self.signal1.x[idx],), (self.signal1.y[idx],))
                plt.scatter((sx,), (sy,))
                plt.savefig("Debug.png")

            square_dim[idx] = dim

        max_idx = np.argmax(np.abs(square_dim))
        self.square_dim = float(square_dim[max_idx])
        self.square_dim_anchor = int(max_idx)
        return self.square_dim, self.square_dim_anchor

    def plot(self, save: bool = False):
        """
        Plots the noise margin square over the two read_waveforms.
        :param save: If true, saves the plot to {self.name}.png
        """
        if self.square_dim == 0:
            self.calculate_square_dim()

        square_x = np.array((
            self.signal1.x[self.square_dim_anchor],
            self.signal1.x[self.square_dim_anchor],
            self.signal1.x[self.square_dim_anchor] + self.square_dim,
            self.signal1.x[self.square_dim_anchor] + self.square_dim,
            self.signal1.x[self.square_dim_anchor],
        ))

        square_y = np.array((
            self.signal1.y[self.square_dim_anchor],
            self.signal1.y[self.square_dim_anchor] + self.square_dim,
            self.signal1.y[self.square_dim_anchor] + self.square_dim,
            self.signal1.y[self.square_dim_anchor],
            self.signal1.y[self.square_dim_anchor]
        ))

        plt.figure(figsize=(6, 6))
        plt.plot(self.signal1.x, self.signal1.y, label=self.signal1.title)
        plt.plot(self.signal2.x, self.signal2.y, label=self.signal2.title)
        plt.plot(square_x, square_y, label=f"Square dim = {self.square_dim}")
        plt.legend()
        plt.title("RSNM Eye Diagram")
        plt.grid(visible=True, which="both", axis="both")
        if save:
            plt.savefig(f"{self.name}.png")
        plt.show()


class ReadSRAMNoiseMarginResultSpecification(DCResultSpecification):
    """
    Specification class that extends DCResultSpecification. Narrows the allowed number of
    read_result signals to 2.
    """
    def __init__(self, signals: List[str]):
        """
        Instantiates the ReadSRAMNoiseMarginResultSpecification.
        Accepts a list of two waveform names.
        """
        assert len(signals) == 2
        super().__init__(signals)

    def interpret(self, waveforms: List[WaveForm], name: str = "RSNM Results") -> ReadSRAMNoiseMarginResult:
        """
        Interprets the read_spec to produce a ReadSRAMNoiseMarginResult.
        :param waveforms: A list of WaveForm objects (that may be a superset of the two signals indicated in the spec).
        :param name: The name of the resulting ReadSRAMNoiseMarginResult.
        :return: A ReadSRAMNoiseMarginResult object.
        """
        result_waveforms: List[WaveForm] = list()
        for waveform in waveforms:
            if waveform.title in self.signals:
                result_waveforms.append(waveform)

        return ReadSRAMNoiseMarginResult(result_waveforms, name)

class WriteSRAMNoiseMarginResult(DCResult):
    """
    Class for calculating the Write SRAM Noise Margin.
    May be extended/split in future versions to cover Hold Noise margin (if
    changes needed)
    """
    def __init__(self, signals: List[WaveForm], name: str = "RSNM Results"):
        """
        Initializes the WriteSRAMNoiseMarginResult.
        :param signals: List of WaveForm objects (ostensibly from spec)
        :param name: Name of the read_result
        """
        if len(signals) != 2:
            raise Exception("This WSNM ResultSpecification class only supports analyzing a pair of signals")
        super().__init__(signals, name)

        if signals[0].x[-1] > signals[1].x[-1]:
            self.signal1 = self.signals[signals[1].title]
            self.signal2 = self.signals[signals[0].title]
        else:
            self.signal1 = self.signals[signals[0].title]
            self.signal2 = self.signals[signals[1].title]
        self.square_dim: float = 0
        self.square_dim_anchor: int = -1

    def reconcile(self):
        """
        Reconciles the two underlying read_waveforms so that their x axes include the same points, allowing direct
        point comparisons.

        Calls the reconcile method from outside the class to do this.
        """
        old_signal1_max = self.signal1.x[-1]
        self.signal1, self.signal2 = reconcile(self.signal1, self.signal2)
        new_signal1_max = np.argwhere((self.signal1.x == old_signal1_max))[0][0] + 1
        self.signal1.y[new_signal1_max:] = np.nan

    def calculate_square_dim(self):
        """
        Calculates the write SRAM noise margin (WSNM). This is done by finding
        the size of the largest nontrivial square that one can inscribe from
        signal1 to signal2 beyond the region of signal2 which is just a
        horizontal line.
        :return: The dimension of the inscribed square.
        """
        y_intercepts = unit_line_through_y_intercept(self.signal1, np.arange(len(self.signal1.x)))
        unit_line_y_value_array = cast_onto_unit_line(self.signal1.x, y_intercepts)
        square_dim = np.ones(len(self.signal1.x)) * -1
        for idx in range(len(self.signal1.x)):
            if np.isnan(self.signal1.y[idx]):
                break
            unit_line_y_values = unit_line_y_value_array[idx, :]
            unit_line = WaveForm(self.signal1.x[idx], unit_line_y_values, "Unit Line")
            unit_line_intercept_idx = argfind_intercept(self.signal2, unit_line)
            if unit_line_intercept_idx == -1:
                continue

            xs = float(self.signal2.x[unit_line_intercept_idx - 1])
            xf = float(self.signal2.x[unit_line_intercept_idx])
            ys = float(self.signal2.y[unit_line_intercept_idx - 1])
            yf = float(self.signal2.y[unit_line_intercept_idx])
            us = float(xs + y_intercepts[idx])
            uf = float(xf + y_intercepts[idx])

            p1 = (xs, ys)
            p2 = (xf, yf)
            p3 = (xs, us)
            p4 = (xf, uf)

            sx, sy = linear_intercept(p1, p2, p3, p4)
            dim = sx - self.signal1.x[idx]
            def debug_plot():
                plt.figure()
                plt.plot(self.signal1.x, self.signal1.y)
                plt.plot(self.signal2.x, self.signal2.y)
                plt.plot(self.signal1.x, unit_line_y_values)
                plt.scatter((self.signal1.x[idx],), (self.signal1.y[idx],))
                plt.scatter((sx,), (sy,))
                plt.savefig("Debug.png")

            square_dim[idx] = dim

        max_idx = np.argmax(square_dim)
        zeros = np.where(square_dim == -1)[0]
        square_dim[zeros] = np.inf
        min_idx = np.argmin(np.abs(square_dim[max_idx:])) + max_idx
        self.square_dim = float(square_dim[min_idx])
        self.square_dim_anchor = int(min_idx)
        return self.square_dim, self.square_dim_anchor

    def plot(self, save: bool = False):
        """
        Plots the noise margin square over the two read_waveforms.
        :param save: If true, saves the plot to {self.name}.png
        """
        if self.square_dim == 0:
            self.calculate_square_dim()

        square_x = np.array((
            self.signal1.x[self.square_dim_anchor],
            self.signal1.x[self.square_dim_anchor],
            self.signal1.x[self.square_dim_anchor] + self.square_dim,
            self.signal1.x[self.square_dim_anchor] + self.square_dim,
            self.signal1.x[self.square_dim_anchor],
        ))

        square_y = np.array((
            self.signal1.y[self.square_dim_anchor],
            self.signal1.y[self.square_dim_anchor] + self.square_dim,
            self.signal1.y[self.square_dim_anchor] + self.square_dim,
            self.signal1.y[self.square_dim_anchor],
            self.signal1.y[self.square_dim_anchor]
        ))

        plt.figure(figsize=(6, 6))
        plt.plot(self.signal1.x, self.signal1.y, label=self.signal1.title)
        plt.plot(self.signal2.x, self.signal2.y, label=self.signal2.title)
        plt.plot(square_x, square_y, label=f"Square dim = {self.square_dim}")
        plt.legend()
        plt.title("WSNM Eye Diagram")
        plt.grid(visible=True, which="both", axis="both")
        if save:
            plt.savefig("{self.name}.png")
        plt.show()

class WriteSRAMNoiseMarginResultSpecification(DCResultSpecification):
    """
    Specification class that extends DCResultSpecification. Narrows the allowed number of
    write_result signals to 2.
    """
    def __init__(self, signals: List[str]):
        """
        Instantiates the WriteSRAMNoiseMarginResultSpecification.
        Accepts a list of two waveform names.
        """
        assert len(signals) == 2
        super().__init__(signals)

    def interpret(self, waveforms: List[WaveForm], name: str = "WSNM Results") -> WriteSRAMNoiseMarginResult:
        """
        Interprets the read_spec to produce a WriteSRAMNoiseMarginResult.
        :param waveforms: A list of WaveForm objects (that may be a superset of the two signals indicated in the spec).
        :param name: The name of the resulting WriteSRAMNoiseMarginResult.
        :return: A WriteSRAMNoiseMarginResult object.
        """
        result_waveforms: List[WaveForm] = list()
        for waveform in waveforms:
            if waveform.title in self.signals:
                result_waveforms.append(waveform)

        return WriteSRAMNoiseMarginResult(result_waveforms, name)