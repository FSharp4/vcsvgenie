from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from vcsvgenie.waveform import WaveForm


class DCResult:

    def __init__(
            self,
            signals: List[WaveForm],
            name: str = "DC Results",
    ):
        self.signals: Dict[str, WaveForm] = dict()
        for signal in signals:
            x = signal.x
            y = signal.y
            sorted_x, x_idxes = np.sort(x), np.argsort(x)
            sorted_y = y[x_idxes]
            self.signals[signal.title] = WaveForm(sorted_x, sorted_y, signal.title)

class DCResultSpecification:
    def __init__(
            self,
            signals: List[str]
    ):
        self.signals = signals

    def interpret(self, waveforms: List[WaveForm], name: str = "DC Results") -> DCResult:
        result_waveforms: List[WaveForm] = list()
        for waveform in waveforms:
            if waveform.title in self.signals:
                result_waveforms.append(waveform)

        return DCResult(result_waveforms, name)

def symmetry_axis_projection(w: WaveForm) -> Tuple[NDArray[np.float64], WaveForm]:
    """
    Written for RSNM calculation
    """
    x = w.x
    y = w.y
    v = y - (1 - x)
    h = x - (1 - y)
    distance = np.sqrt(np.square(v) + np.square(h))
    sx = x - h
    sy = y - v
    projection = WaveForm(sx, sy, f"proj({w.title})")
    return distance, projection

def reconcile(w1: WaveForm, w2: WaveForm) -> Tuple[WaveForm, WaveForm]:
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
    w1_is_upper = w1.y >= w2.y
    dw1_is_upper = np.bitwise_xor(w1_is_upper[:-1], w1_is_upper[1:])
    transition = int(np.argwhere(dw1_is_upper)[0] + 1)
    return transition

def argfind_eye(w1: WaveForm, w2: WaveForm) -> Tuple[int, int]:
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
    """Calculates b for line(s) y = x + b intercepting the waveform w at indexes idx"""
    x = w.x[idx]
    y = w.y[idx]
    return y - x


def cast_onto_unit_line(x: NDArray[np.float64], b: NDArray[np.float64] | np.float64) -> NDArray[np.float64]:
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
    l1m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    l1b = p1[1] - l1m * p1[0]
    l2m = (p4[1] - p3[1]) / (p4[0] - p3[0])
    l2b = p3[1] - l2m * p3[0]
    intercept_x = (l1b - l2b) / (l2m - l1m)
    intercept_y = l1m * intercept_x + l1b
    return intercept_x, intercept_y


class ReadSRAMNoiseMarginResult(DCResult):
    def __init__(self, signals: List[WaveForm], name: str = "RSNM Results"):
        if len(signals) != 2:
            raise Exception("This RSNM ResultSpecification class only supports analyzing a pair of signals")
        super().__init__(signals, name)

        self.signal1 = self.signals[signals[0].title]
        self.signal2 = self.signals[signals[1].title]
        self.square_dim: float = 0
        self.square_dim_anchor: int = -1

    def truncate(self):
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
        self.signal1, self.signal2 = reconcile(self.signal1, self.signal2)

    def calculate_square_dim(self) -> Tuple[float, int]:
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

    def plot(self):
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
        plt.show()


class ReadSRAMNoiseMarginResultSpecification(DCResultSpecification):
    def __init__(self, signals: List[str]):
        assert len(signals) == 2
        super().__init__(signals)

    def interpret(self, waveforms: List[WaveForm], name: str = "RSNM Results") -> ReadSRAMNoiseMarginResult:
        result_waveforms: List[WaveForm] = list()
        for waveform in waveforms:
            if waveform.title in self.signals:
                result_waveforms.append(waveform)

        return ReadSRAMNoiseMarginResult(result_waveforms, name)

