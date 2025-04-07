from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Literal, Tuple
from warnings import warn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sortedcontainers import SortedDict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame

from vcsvgenie.waveform import WaveForm, linear_interpolation


@dataclass
class Bus:
    """
    Represents a multi-bit bus
    """
    title: str
    members: List[str]
    powers: List[int]

    def equals(self, bus: "Bus"):
        """
        Returns True if the bus equals the given bus
        """
        if self.title != bus.title:
            return False
        if self.members != bus.members:
            return False
        if self.powers != bus.powers:
            return False

        return True


@dataclass
class Transition:
    """
    Represents a single transition in a single waveform
    :param transition_type: Either "Rising" or "Falling"
    :param series: The title of the waveform
    :param series_type: Whether the waveform is an input signal or an output signal
    :param interval: The periodic interval in which the transition occurred
    :param transition_time: The amount of time needed to produce the transition
    :param low_threshold: The low threshold of the transition
    :param high_threshold: The high threshold of the transition
    """
    transition_type: Literal["Rising", "Falling"]
    series: str
    series_type: Literal["Input", "Output"]
    interval: int
    transition_time: float
    low_threshold: float = 0.1
    high_threshold: float = 0.9

    def __str__(self) -> str:
        """
        Returns a string representation of the transition
        """
        return f"i={self.interval} [{self.series_type}] {self.series} {self.transition_type} transition observed"


@dataclass
class Propagation:
    """
    Represents a single propagation between two read_waveforms (one input and one output)
    One input waveform that experiences a transition may cause an output waveform to experience a transition shortly
    after.

    :param source: The input waveform title.
    :param destination: The output waveform title.
    :param departure: The timestamp (in seconds) of the first/input waveform transition (50% logic threshold).
    :param arrival: The timestamp (in seconds) of the second/output waveform transition (50% logic threshold).
    :param propogation_type: Either "Rising" or "Falling". Determined by the type of the transition at the output.
    :param delay: The propogation delay (the arrival timestamp minus the departure timestamp).
    :param interval: The periodic interval in which the propagation occurred.
    """
    source: str
    destination: str
    departure: float
    arrival: float
    propagation_type: Literal["Rising", "Falling"]
    delay: float
    interval: int

    def type_label(self) -> str:
        """Returns a string summary of the propagation (identifying which read_waveforms are involved)"""
        return f"{self.source} -> {self.destination}"

    def __str__(self) -> str:
        """Returns a string representation of the propagation"""
        return f"[i={self.interval}] {self.source} -> {self.destination} ({self.propagation_type}): {self.delay}"


class TransientResult:
    """
    Represents a collection of results from a single transient analysis run.
    """
    def __init__(
            self,
            inputs: List[WaveForm],
            outputs: List[WaveForm],
            name: str = "Transient Results",
            input_bus_dict: Dict | None = None,
            output_bus_dict: Dict | None = None,
            clock_period: float = 1e-9,
            logic_threshold: float = 0.5,
            absolute_bus_bits: bool = True,
            bus_display_radix: Literal["Unsigned Decimal"] = "Unsigned Decimal",
    ):
        """
        Initializes the TransientResult.
        :param inputs: The input read_waveforms
        :param outputs: The output read_waveforms
        :param name: The name of the read_result
        :param input_bus_dict: A dictionary mapping names of input bus objects to the input bus objects
        :param output_bus_dict: A dictionary mapping names of output bus objects to the output bus objects
        :param clock_period: The clock period (in seconds)
        :param logic_threshold: The logic threshold (in volts)
        :param absolute_bus_bits: Whether the input and output bus objects are contiguous streams of bits
               (aka, their lengths can be inferred by looking at the indice of the highest bit)
        :param bus_display_radix: Defines how bus values should be interpreted when displayed to user.
        """
        if output_bus_dict is None:
            output_bus_dict = {}
        if input_bus_dict is None:
            input_bus_dict = {}
        self.timestamps = inputs[0].x
        self.inputs: Dict[str, NDArray[np.float64]] = dict()
        self.outputs: Dict[str, NDArray[np.float64]] = dict()
        self.input_bus_spec = input_bus_dict
        self.output_bus_spec = output_bus_dict
        self.absolute_bus_bits = absolute_bus_bits
        self.bus_display_radix = bus_display_radix
        for _input in inputs:
            if not np.allclose(_input.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.inputs[_input.title] = _input.y

        for output in outputs:
            if not np.allclose(output.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.outputs[output.title] = output.y

        self.name = name
        self.clock_period = clock_period
        self.eps_n_timestamps = 10
        self.LOGIC_THRESHOLD = logic_threshold
        self._transitions: List[List[Transition]] = list()
        self._interval_start_idxs: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self._interval_end_idxs: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self._propagations: List[Propagation] = list()

        self.n_intervals = int(
            np.ceil((self.timestamps[-1] - self.timestamps[0]) / self.clock_period)
        )
        self.digital_inputs: Dict[str, NDArray[np.bool_]] = dict()
        self.digital_outputs: Dict[str, NDArray[np.bool_]] = dict()
        self.input_bus_values: Dict[str, NDArray[np.int64]] = dict()
        self.output_bus_values: Dict[str, NDArray[np.int64]] = dict()

    @property
    def propagations(self) -> List[Propagation]:
        """
        List of propagation objects.
        If an Exception is thrown, this means this list has not been compiled yet.
        """
        if len(self._propagations) == 0:
            raise Exception(
                "Propagations not calculated (call TransientResult.find_propagations first)"
            )

        return self._propagations

    @property
    def delays(self) -> NDArray[np.float64]:
        """
        Array of propagation objects
        """
        return np.array([propagation.delay for propagation in self.propagations], dtype=np.float64)

    @property
    def sorted_delays(self) -> NDArray[np.float64]:
        """
        Sorted array of propagation objects
        """
        return np.array(
            [
                propagation.delay for propagation in sorted(
                self.propagations, key=lambda propagation: propagation.delay
            )
            ], dtype=np.float64
        )

    @property
    def transitions(self) -> List[List[Transition]]:
        """
        List of transition objects.
        If an Exception is thrown this means the list has not been compiled yet.
        """
        if len(self._transitions) == 0:
            raise Exception(
                "Transitions not calculated (call TransientResult.find_transitions first)"
            )

        return self._transitions

    def propagations_by_interval(self) -> List[List[Propagation]]:
        """
        Propagation objects, partitioned by periodic interval in a list of lists
        """
        propagations = self.propagations
        interval_propagations: List[List[Propagation]] = [[] for _ in range(self.n_intervals)]

        for propagation in propagations:
            interval_propagations[propagation.interval].append(propagation)

        return interval_propagations

    def max_delay_by_interval(self) -> NDArray[np.float64]:
        """
        Array of maximum propagation delays (in seconds) occurring over each interval
        """
        interval_propagations = self.propagations_by_interval()
        max_delays = np.zeros(self.n_intervals, dtype=np.float64)
        for idx, propagations in enumerate(interval_propagations):
            if len(propagations) == 0:
                continue

            max_delays[idx] = max([propagation.delay for propagation in propagations])

        return max_delays

    @property
    def interval_start_idxs(self) -> NDArray[np.int64]:
        """
        An array of the timestamp idxes indicating the start of each periodic interval
        If an Exception is thrown, this array has not been compiled yet.
        """
        if np.all(self._interval_start_idxs == 0):
            raise Exception(
                "Interval Start Timestamp Indexes not calculated (call TransientResult.find_transitions first)"
            )

        return self._interval_start_idxs

    @property
    def interval_end_idxs(self) -> NDArray[np.int64]:
        """
        An array of the timestamp idxes indicating the end of each periodic interval
        If an Exception is thrown, this array has not been compiled yet.
        """
        if np.all(self._interval_end_idxs == 0):
            raise Exception(
                "Interval End Timestamp Indexes not calculated (call TransientResult.find_transitions first)"
            )

        return self._interval_end_idxs

    def plot(
            self, save: bool = False, display: bool = True, separate: bool = False
    ) -> None:
        """
        Creates a plot of the transient read_result object.
        This is more useful for objects that don't include too many read_waveforms.
        :param save: Whether to save the plot to disk
        :param display: Whether to display the plot immediately
        :param separate: Whether to separate the read_waveforms into separate plots
        """
        if not display and not save:
            warn(
                "TransientResult.plot called without save nor display; defaulting to display"
            )
            display = True
        if not separate:
            plt.figure()
            for title in self.inputs:
                plt.plot(self.timestamps, self.inputs[title], label=title)
            for title in self.outputs:
                plt.plot(self.timestamps, self.outputs[title], label=title)
            plt.xlabel("Timestamp (ns)")
            plt.ylabel("Voltage (V)")
            plt.title(self.name)
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
            if display:
                plt.show()
            else:
                plt.savefig(f"TransientResult__plot {self.name}.png")
                plt.close()
        else:
            for title in self.inputs:
                plt.figure()
                plt.plot(self.timestamps, self.inputs[title])
                plt.xlabel("Timestamp (s)")
                plt.ylabel("Voltage (V)")
                plt.title(f"{self.name} {title}")
                plt.grid(visible=True, which="both", axis="both")
                if display:
                    plt.show()
                else:
                    plt.savefig(
                        f"TransientResult__plot {self.name}_{title.replace('/', '_')}.png"
                    )
                    plt.close()

            for title in self.outputs:
                plt.figure()
                plt.plot(self.timestamps, self.outputs[title], "r")
                plt.xlabel("Timestamp (s)")
                plt.ylabel("Voltage (V)")
                plt.title(f"{self.name} {title}")
                plt.grid(visible=True, which="both", axis="both")
                if display:
                    plt.show()
                else:
                    plt.savefig(
                        f"TransientResult__plot {self.name}_{title.replace('/', '_')}.png"
                    )
                    plt.close()

    def find_transitions(self, eps_n_timestamps: int = 1, low_threshold: float = 0.1,
                         high_threshold: float = 0.9) -> None:
        """
        Finds all logic transitions in the read_waveforms
        :param eps_n_timestamps: Number of timestamps to subtract from the end of the interval (to remove potential skew issues)
        :param low_threshold: Lower bound of the transitions to produce
        :param high_threshold: Upper bound of the transitions to produce
        """
        self.eps_n_timestamps = eps_n_timestamps
        ending_timestamp = self.timestamps[-1]

        start_timestamp_idxs = np.zeros(self.n_intervals, dtype=np.int64)
        end_timestamp_idxs = np.zeros(self.n_intervals, dtype=np.int64)
        transitions: List[List[Transition]] = list()
        for n in range(self.n_intervals):
            start = self.clock_period * n
            end = self.clock_period * (n + 1)
            start_timestamp_idxs[n] = np.searchsorted(
                self.timestamps, start, side="right"
            )
            end_timestamp_idxs[n] = (
                    np.searchsorted(self.timestamps, end, side="right") - eps_n_timestamps
            )
            transitions.append([])

        def process_transition(
            _idx: int,
            _start_timestamp_idxs: NDArray[np.int64],
            _end_timestamp_idxs: NDArray[np.int64]
        ) -> Tuple[Literal["Rising", "Falling"], float]:
            """
            Determines the orientation (Rising/Falling) and the transition time of the transition
            :param _idx: The index of the periodic interval in which the transition occurs
            :param _start_timestamp_idxs: The timestamp idxes of the start of the transition
            :param _end_timestamp_idxs: The timestamp idxes of the end of the transition
            """
            interval_data = data[_start_timestamp_idxs[_idx]:_end_timestamp_idxs[_idx]]
            _transition_type: Literal["Rising", "Falling"] = "Rising" if ends_data[_idx] else "Falling"

            def prep_interpolation(pivot_idx: int, threshold) -> float:
                """
                Prepares for and calls linear_interpolation on a line segment.
                Used for determining the exact point at which a transition threshold is crossed.
                :param pivot_idx: The beginning index of the line segment for which to run interpolation on
                :param threshold: y=threshold comparison to run interpolation to find crossing point with respect to
                """
                increment = -1 if _transition_type == "Rising" else 1
                x1: float = float(self.timestamps[_start_timestamp_idxs[_idx] + pivot_idx])
                x0: float = float(self.timestamps[_start_timestamp_idxs[_idx] + pivot_idx - 1])
                y0: float = float(data[_start_timestamp_idxs[_idx] + pivot_idx - 1])
                y1: float = float(data[_start_timestamp_idxs[_idx] + pivot_idx])
                timestamp = linear_interpolation(x0, x1, y0, y1, threshold, 'y')
                return timestamp

            def extract_transition_time(_low_idx: int, _high_idx: int) -> float:
                """
                Extracts the exact time of the transition between two timestamp indexes which indicate when the
                transition occurs.
                :param _low_idx: The beginning index of the line segment for which to run interpolation on for the
                                 low threshold
                :param _high_idx: The beginning index of the line segment for which to run interpolation on for the
                                  high threshold
                """
                low_timestamp = self.timestamps[_low_idx + _start_timestamp_idxs[_idx]]
                if _low_idx > 0:
                    low_timestamp = prep_interpolation(_low_idx, low_threshold)

                high_timestamp = self.timestamps[_high_idx + _start_timestamp_idxs[_idx]]
                if _high_idx > 0:
                    high_timestamp = prep_interpolation(_high_idx, high_threshold)

                __transition_time = high_timestamp - low_timestamp
                return __transition_time

            try:
                if _transition_type == "Rising":
                    low_idx: int = int(np.where(interval_data > low_threshold)[0][0])
                    high_idx: int = int(np.where(interval_data > high_threshold)[0][0])
                else:  # Falling
                    high_idx: int = int(np.where(interval_data < high_threshold)[0][0])
                    low_idx: int = int(np.where(interval_data < low_threshold)[0][0])
            except IndexError as e:
                # Invalid transition
                raise e

            try:
                _transition_time: float = extract_transition_time(low_idx, high_idx)
                return _transition_type, _transition_time
            except IndexError as e:
                # Invalid transition
                raise e

        for input_series in self.inputs.keys():
            data = self.inputs[input_series]
            starts_data = data[start_timestamp_idxs] > self.LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > self.LOGIC_THRESHOLD

            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    try:
                        transition_type, transition_time = process_transition(idx, start_timestamp_idxs, end_timestamp_idxs)
                    except IndexError:
                        # Invalid transition
                        continue # TODO: Find a better solution for intervals who start on the same side of the logic threshold as their supposed transition polarity
                    transition = Transition(
                        transition_type,
                        input_series,
                        "Input",
                        idx,
                        transition_time,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold
                    )

                    transitions[idx].append(transition)

        for output_series in self.outputs.keys():
            data = self.outputs[output_series]
            starts_data = data[start_timestamp_idxs] > self.LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > self.LOGIC_THRESHOLD

            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    try:
                        transition_type, transition_time = process_transition(idx, start_timestamp_idxs, end_timestamp_idxs)
                    except IndexError:
                        # Invalid transition
                        continue
                    transition = Transition(
                        transition_type,
                        output_series,
                        "Output",
                        idx,
                        transition_time,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold
                    )
                    transitions[idx].append(transition)

        self._transitions = transitions
        self._interval_start_idxs = start_timestamp_idxs
        self._interval_end_idxs = end_timestamp_idxs

    def find_propagations(self) -> None:
        """
        Finds all propagations that occur within the WaveForms in this Result
        """
        propagations: List[Propagation] = list()
        for idx, timestep_transitions in enumerate(self.transitions):
            if len(timestep_transitions) < 2:
                continue

            input_transitions: List[Transition] = [
                transition
                for transition in timestep_transitions
                if transition.series_type == "Input"
            ]
            output_transitions: List[Transition] = [
                transition
                for transition in timestep_transitions
                if transition.series_type == "Output"
            ]
            if len(output_transitions) == 0:
                continue

            for input_transition in input_transitions:
                input_transition_timestamp = self.interpolate_transition_timestamp(
                    self.LOGIC_THRESHOLD, input_transition
                )
                for output_transition in output_transitions:
                    output_transition_timestamp = self.interpolate_transition_timestamp(
                        self.LOGIC_THRESHOLD, output_transition
                    )
                    propagation_delay = (
                            output_transition_timestamp - input_transition_timestamp
                    )
                    propagations.append(
                        Propagation(
                            input_transition.series,
                            output_transition.series,
                            input_transition_timestamp,
                            output_transition_timestamp,
                            output_transition.transition_type,
                            propagation_delay,
                            input_transition.interval,
                        )
                    )

        self._propagations = propagations
        # return propagations

    def digitize(self):
        """Digitizes the waveform data into 1s and 0s per the logic threshold."""
        for input_name in self.inputs:
            if self.inputs[input_name][0] < self.LOGIC_THRESHOLD:
                self.digital_inputs[input_name] = np.zeros(
                    (self.n_intervals), dtype=np.bool_
                )
            else:
                self.digital_inputs[input_name] = np.ones(
                    (self.n_intervals), dtype=np.bool_
                )

        for output_name in self.outputs:
            if self.outputs[output_name][0] < self.LOGIC_THRESHOLD:
                self.digital_outputs[output_name] = np.zeros(
                    (self.n_intervals), dtype=np.bool_
                )
            else:
                self.digital_outputs[output_name] = np.ones(
                    (self.n_intervals), dtype=np.bool_
                )

        for interval, transitions in enumerate(self.transitions):
            if len(transitions) == 0:
                continue

            for transition in transitions:
                signal = transition.series
                signal_type = transition.series_type
                to_value = False
                if transition.transition_type == "Rising":
                    to_value = True

                if signal_type == "Input":
                    self.digital_inputs[signal][interval:] = to_value
                else:
                    self.digital_outputs[signal][interval:] = to_value

    def resolve_buses(self):
        """
        Resolves the bus values for all buses defined in this Result (both input and output).
        Useful for functional verification.
        """
        for bus_name in self.input_bus_spec:
            input_bus = self.input_bus_spec[bus_name]
            signals = input_bus.members
            powers = input_bus.powers
            if self.absolute_bus_bits == False:
                powers = list(range(len(powers) - 1, -1, -1))

            self.input_bus_values[bus_name] = np.zeros((self.n_intervals), dtype=np.int64)
            for interval in range(self.n_intervals):
                value: int = 0
                for idx, signal in enumerate(signals):
                    value += self.digital_inputs[signal][interval] * (1 << powers[idx])

                self.input_bus_values[bus_name][interval] = value

        for bus_name in self.output_bus_spec:
            output_bus = self.output_bus_spec[bus_name]
            signals = output_bus.members
            powers = output_bus.powers
            if self.absolute_bus_bits == False:
                powers = list(range(len(powers) - 1, -1, -1))

            self.output_bus_values[bus_name] = np.zeros((self.n_intervals), dtype=np.int64)
            for interval in range(self.n_intervals):
                value: int = 0
                for idx, signal in enumerate(signals):
                    value += self.digital_outputs[signal][interval] * (1 << powers[idx])

                self.output_bus_values[bus_name][interval] = value

    def tabulate_bus_data(self):
        """
        Presents all input/output bus data in a pandas read_dataframe.
        :return: The tabulated bus data in a DataFrame
        """
        input_table = DataFrame.from_dict(self.input_bus_values)
        output_table = DataFrame.from_dict(self.output_bus_values)
        table = pd.concat([input_table, output_table], axis=1)
        return table

    def interpolate_transition_timestamp(
            self, LOGIC_THRESHOLD: float, transition: Transition
    ) -> float:
        """
        Does linear interpolation to determine when transitions between logic low and logic high values occur.
        :param LOGIC_THRESHOLD: Threshold for the logic low value
        :param transition: Transition to interpolate
        """
        sweep_start_timestamp: np.int64 = self.interval_start_idxs[transition.interval]
        sweep_end_timestamp: np.int64 = self.interval_end_idxs[transition.interval] + 1
        data_interval: NDArray[np.float64]
        if transition.series_type == "Input":
            data_interval = self.inputs[transition.series][
                            sweep_start_timestamp:sweep_end_timestamp
                            ]
        else:
            data_interval = self.outputs[transition.series][
                            sweep_start_timestamp:sweep_end_timestamp
                            ]

        check_transitioned = data_interval > LOGIC_THRESHOLD
        if check_transitioned[0]:
            check_transitioned = np.logical_not(check_transitioned)

        deltastamp: int = 0
        for idx, is_transitioned in enumerate(check_transitioned):
            if not is_transitioned:
                deltastamp = idx

        x_low: int = sweep_start_timestamp + deltastamp
        x_high: int = x_low + 1
        y_low: np.float64 = data_interval[deltastamp]
        y_high: np.float64 = data_interval[deltastamp + 1]
        proportion: float = (LOGIC_THRESHOLD - y_low) / (y_high - y_low)
        time_low = self.timestamps[x_low]
        time_high = self.timestamps[x_high]
        interpolated_timestamp: float = time_low + (time_high - time_low) * proportion
        return interpolated_timestamp


class BusValidationException(Exception):
    """Exception subclass which also notes where the error occurs in a bus"""
    def __init__(self, bus_title: str, signal_title: str, *args):
        """
        Initializes the BusValidationException.
        :param bus_title: The name of the bus
        :param signal_title: The name of the signal within the bus
        """
        super.__init__(*args)
        self.bus_title = bus_title
        self.signal_title = signal_title


# @dataclass
class TransientResultSpecification:
    """
    Specification for a TransientResult.
    Useful for producing TransientResult objects over multiple analysis runs.
    """
    def __init__(
            self,
            inputs: List[str],
            outputs: List[str],
            input_buses: Dict[str, Bus] | None = None,
            output_buses: Dict[str, Bus] | None = None,
            logic_threshold: float = 0.5,
            clock_period: float = 1e-9,
    ):
        """
        Initializes the TransientResultSpecification.

        One caveat: Bidirectional signals which are referenced as both inputs and outputs have not been tested.
        For a surefire workaround, define them as one or the other and do your own work to figure out propagation
        timings where these signals are the other type.

        :param inputs: The input signal read_titles
        :param outputs: The output signal read_titles.
        :param input_buses: An input bus read_spec (a dictionary mapping the names of the buses to the buses)
        :param output_buses: An output bus read_spec (a dictionary mapping the names fo the buses to the buses)
        :param logic_threshold: The logic threshold (in Volts)
        :param clock_period: The clock period (in seconds)
        """
        self.inputs = inputs
        self.outputs = outputs
        self.input_buses = input_buses if input_buses is not None else {}
        self.output_buses = output_buses if output_buses is not None else {}
        self.logic_threshold = logic_threshold
        self.clock_period = clock_period

    def infer_buses(self) -> None:
        """
        Infers the existence of buses from signals which follow Cadence bus notation

        EX:
        a<3:0> has signals a<3>, a<2>, a<1>, a<0>

        This method will gather these signals up into a bus (with type of either input or output) even if the bus is
        not present in spec. (It will augment the spec to accommodate the bus).

        Do not run this method if, for whatever reason, you have signals that follow bus notations but some are inputs
        and some are outputs.
        """
        unique_input_signal_collectors: Dict[str, SortedDict[int, str]] = {}
        unique_output_signal_collectors: Dict[str, SortedDict[int, str]] = {}
        for input in self.inputs:
            tokens = input.split("<")
            if len(tokens) != 2:
                continue

            numeric = int(tokens[1][:-1])
            bus_name = tokens[0]
            if bus_name not in unique_input_signal_collectors.keys():
                unique_input_signal_collectors[bus_name] = {}

            unique_input_signal_collectors[bus_name][numeric] = input

        for output in self.outputs:
            tokens = output.split("<")
            if len(tokens) != 2:
                continue

            numeric = int(tokens[1][:-1])
            bus_name = tokens[0]
            if bus_name not in unique_output_signal_collectors.keys():
                unique_output_signal_collectors[bus_name] = {}

            unique_output_signal_collectors[bus_name][numeric] = output

        input_bus_dict: Dict[str, Bus] = {}
        output_bus_dict: Dict[str, Bus] = {}
        for input_bus_name in unique_input_signal_collectors.keys():
            signals: List[str] = []
            signal_dict = unique_input_signal_collectors[input_bus_name]
            keys = [key for key in signal_dict.keys()]
            for index in keys:
                signals.append(signal_dict[index])

            bus = Bus(input_bus_name, signals, keys)
            input_bus_dict[input_bus_name] = bus

        for output_bus_name in unique_output_signal_collectors.keys():
            signals: List[str] = []
            signal_dict = unique_output_signal_collectors[output_bus_name]
            keys = [key for key in signal_dict.keys()]
            for index in keys:
                signals.append(signal_dict[index])

            bus = Bus(output_bus_name, signals, keys)
            output_bus_dict[output_bus_name] = bus

        for key in input_bus_dict.keys():
            if key in self.input_buses:
                if self.input_buses[key].equals(input_bus_dict[key]):
                    continue
                else:
                    self.input_buses[f"{key}_AUTOGEN"] = input_bus_dict[key]
            else:
                self.input_buses[key] = input_bus_dict[key]

        for key in output_bus_dict.keys():
            if key in self.output_buses:
                if self.output_buses[key].equals(output_bus_dict[key]):
                    continue
                else:
                    self.output_buses[f"{key}_AUTOGEN"] = output_bus_dict[key]
            else:
                self.output_buses[key] = output_bus_dict[key]

    def verify_buses(self, error_on_fail: bool = False) -> bool:
        """
        Checks to see if the signals indicated in the bus are present in the spec.

        :param error_on_fail: If True, will raise an exception if any of the signals indicated in a bus are not present
                              in the spec. You will need to set this to True if you want to know which buses are
                              failing.
        :return: True if all the buses pass. Otherwise False.
        """
        for key in self.input_buses:
            bus = self.input_buses[key]
            for signal in bus.members:
                if signal in self.inputs:
                    continue
                else:
                    if error_on_fail:
                        raise BusValidationException(bus.title, signal.title)
                    return False

        for key in self.output_buses:
            bus = self.output_buses[key]
            for signal in bus.members:
                if signal in self.outputs:
                    continue
                else:
                    if error_on_fail:
                        raise BusValidationException(bus.title, signal.title)
                    return False

        return True

    def interpret(self, waveforms: List[WaveForm], name: str = "Transient Results") -> TransientResult:
        """
        Interprets a collection of read_waveforms into a TransientResult.
        :param waveforms: A collection of read_waveforms to interpret
        :param name: The name of the read_result
        :return: A TransientResult
        """
        input_waveforms = []
        output_waveforms = []
        for waveform in waveforms:
            if waveform.title in self.inputs:
                input_waveforms.append(waveform)
            elif waveform.title in self.outputs:
                output_waveforms.append(waveform)

        if self.verify_buses():
            return TransientResult(
                input_waveforms,
                output_waveforms,
                name=name,
                input_bus_dict=self.input_buses,
                output_bus_dict=self.output_buses,
                logic_threshold=self.logic_threshold,
                clock_period=self.clock_period
            )
        else:
            warn("Invalid buses, not registering (run verify_buses(error_on_fail=True) for blame)")
            return TransientResult(
                input_waveforms,
                output_waveforms,
                name=name,
                logic_threshold=self.logic_threshold,
                clock_period=self.clock_period
            )


def average_propagation_delays_by_category(
        propagations: List[Propagation],
) -> Dict[str, Tuple[float, float]]:
    """
    Finds the average propagation delay for each category of propagation (i.e., each unique pair of input-output
    read_waveforms which saw transitions in the same period).
    :param propagations: List of propagations to process
    :return: Dictionary of average propagation delay for each category of propagation
    """
    propagation_dictionary: Dict[str, Tuple[List[Propagation], List[Propagation]]] = (
        dict()
    )
    averages: Dict[str, Tuple[float, float]] = dict()
    for propagation in propagations:
        label = propagation.type_label()
        if label not in propagation_dictionary.keys():
            propagation_dictionary[label] = ([], [])

        if propagation.propagation_type == "Rising":
            propagation_dictionary[label][0].append(propagation)
        else:
            propagation_dictionary[label][1].append(propagation)

    for category in propagation_dictionary.keys():
        rising_propagations, falling_propagations = propagation_dictionary[category]
        if len(rising_propagations) == 0 or len(falling_propagations) == 0:
            continue

        average_rising_delay: float = 0
        for rising_propagation in rising_propagations:
            average_rising_delay += rising_propagation.delay

        average_rising_delay /= len(rising_propagations)

        average_falling_delay: float = 0
        for falling_propagation in falling_propagations:
            average_falling_delay += falling_propagation.delay

        average_falling_delay /= len(falling_propagations)

        averages[category] = (average_rising_delay, average_falling_delay)

    return averages


def maximum_propagation_delays_by_category(
        propagations: List[Propagation],
) -> Dict[str, Tuple[Propagation, Propagation]]:
    """
    Finds the maximum propagation delay for each category of propagation (i.e., each unique pair of input-output
    read_waveforms which saw transitions in the same period).
    :param propagations: List of propagations to process
    :return: Dictionary of maximum propagation delay for each category of propagation
    """
    propagation_dictionary: Dict[str, Tuple[List[Propagation], List[Propagation]]] = (
        dict()
    )
    maxima: Dict[str, Tuple[Propagation, Propagation]] = dict()
    for propagation in propagations:
        label = propagation.type_label()
        if label not in propagation_dictionary.keys():
            propagation_dictionary[label] = ([], [])

        if propagation.propagation_type == "Rising":
            propagation_dictionary[label][0].append(propagation)
        else:
            propagation_dictionary[label][1].append(propagation)

    for category in propagation_dictionary.keys():
        rising_propagations, falling_propagations = propagation_dictionary[category]
        if len(rising_propagations) == 0 or len(falling_propagations) == 0:
            continue

        maximum_rising_delay: float = rising_propagations[0].delay
        maximum_rising_propagation: Propagation = rising_propagations[0]
        for rising_propagation in rising_propagations:
            if rising_propagation.delay > maximum_rising_delay:
                maximum_rising_delay = rising_propagation.delay
                maximum_rising_propagation = rising_propagation

        maximum_falling_delay: float = falling_propagations[0].delay
        maximum_falling_propagation: Propagation = falling_propagations[0]
        for falling_propagation in falling_propagations:
            # maximum_falling_delay += falling_propagation.delay
            if falling_propagation.delay > maximum_falling_delay:
                maximum_falling_delay = falling_propagation.delay
                maximum_falling_propagation = falling_propagation

        maxima[category] = (maximum_rising_propagation, maximum_falling_propagation)

    return maxima


def critical_propagation_delays(propagations: List[Propagation]) -> Tuple[Propagation, Propagation]:
    """
    Finds the longest rising and falling propagation delays.
    :param propagations: List of propagations to process
    :return: The rising and falling propagation delays whose delays are larger than all others of their polarity.
    """
    maximum_rising_delay: float = 0
    rising_blame: Propagation
    maximum_falling_delay: float = 0
    falling_blame: Propagation
    if len(propagations) == 0:
        raise Exception("No propagations registered")

    for propagation in propagations:
        if propagation.propagation_type == "Rising":
            if propagation.delay > maximum_rising_delay:
                maximum_rising_delay = propagation.delay
                rising_blame = propagation
        elif propagation.delay > maximum_falling_delay:
            maximum_falling_delay = propagation.delay
            falling_blame = propagation

    return rising_blame, falling_blame


def quasicritical_propagation_delays(propagations: List[Propagation], samples) -> Tuple[Propagation, ...]:
    """
    Finds the worst propagation delays
    :param propagations: List of propagations to process
    :param samples: Number of worst propagation delays to extract
    :return: The longest propagation delays in a list
    """
    sorted_delays = sorted(propagations, key=lambda propagation: propagation.delay)
    return sorted_delays[-samples:]


def extract_paths(propagations: List[Propagation]) -> List[Tuple[str, str]]:
    """
    Extracts the set of unique paths from a list of propagations.
    :param propagations: List of propagations to process
    :return: A list of tuples containing the path source net and destination net
    """
    unique_paths: List[Tuple[str, str]] = []
    for propagation in propagations:
        path = (propagation.source, propagation.destination)
        if path in unique_paths:
            continue

        unique_paths.append(path)

    return unique_paths


def delay_histogram(delays: NDArray[np.float64], n_bins: int = 100, show=False) -> Figure:
    """
    Creates a histogram of delay timings.
    :param delays: Array of delay timings
    :param n_bins: Number of bins
    :param show: Whether to immediately show the histogram
    :return: Figure of delay histogram.
    """
    hist, bins = np.histogram(delays, bins=n_bins)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5

    figure: Figure = plt.figure()
    plt.plot(bin_centers, hist)
    plt.xlabel("Propagation delay (s)")
    plt.ylabel("Relative frequency of occurrence")
    plt.title("Histogram of propagation delay observations")
    plt.grid(visible=True, which='both', axis='both')
    if show:
        plt.show()
    return figure


def find_max_delay_trend(delays: NDArray[np.float64]) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Calculates the upper envelope of delays occurring in an array of delays.
    Useful for observing how maximum observed delay scales with sample size (an indirect relation to the length of the
    simulation/number of transitions tested)
    :param delays: Array of delay timings
    :return: Tuple containing the upper envelope of delays occurring in an array of delays
    """
    # This code is messy but works. Low priority to fix.
    max_delay_trend: List[np.float64] = [delays[0]]
    max_delay_idxes: List[np.int64] = [0]
    for idx, delay in enumerate(delays):
        if delay > max_delay_trend[-1]:
            max_delay_trend.append(delay)
            max_delay_idxes.append(idx)

    return np.array(max_delay_idxes, dtype=np.int64), np.array(max_delay_trend, dtype=np.float64)


def plot_max_delay_trend(max_delay_idxes: NDArray[np.int64], max_delay_trend: NDArray[np.float64], show=False):
    """
    Plots the maximum delay trend.
    :param max_delay_idxes: Array of indices corresponding to maximum delay trend indexes within the larger array (not
                            included in method)
    :param max_delay_trend: Array of delays corresponding to maximum delay trend indexes
    :param show: Whether to show the plot
    :return: The plot figure
    """
    figure = plt.figure()
    plt.plot(max_delay_idxes, max_delay_trend, linestyle='--', marker='o')
    plt.title("Maximum delay trend over # of propagation delays")
    plt.xlabel("Propagation delay index")
    plt.ylabel("Propgation delay (s)")
    plt.grid(visible=True, which='both', axis='both')
    if show:
        plt.show()
    return figure


def plot_inverse_max_delay_trend(max_delay_idxes: NDArray[np.int64], max_delay_trend: NDArray[np.float64],
                                 show=False) -> Figure:
    """
    Plots the inverse maximum delay trend.
    Done to perform asymptotic analysis on the trend to forecast how this trend will evolve as sample size increases.
    :param max_delay_idxes: Array of indices corresponding to maximum delay trend indexes
    :param max_delay_trend: Array of delays corresponding to maximum delay trend indexes
    :param show: Whether to show the plot
    :return: The max delay trend figure
    """
    figure: Figure = plt.figure()
    plt.plot(1 / np.array(max_delay_idxes, np.float64), max_delay_trend, linestyle='--', marker='o')
    plt.title("Maximum delay trend over (# of propagation delays)^-1")
    plt.xlabel("Propagation delay index inverse")
    plt.ylabel("Propgation delay (s)")
    plt.grid(visible=True, which='both', axis='both')
    if show:
        plt.show()
    return figure


def linear_regression_intercept(invns: NDArray[np.float64], trend: NDArray[np.float64],
                                INV_THRESH: np.float64 = 0) -> np.float64:
    """
    Calculates the x-intercept of the inverse max delay trend.
    :param invns: Array of inverse maximum delay trend values
    :param trend: Array of maximum delay trend values
    :param INV_THRESH: Threshold y=T to calculate x-intercepts with. Set to >0 to simulate a maximum sample size less
                       than infinity.
    :return: x-intercept (indicating maximum delay trend)
    """
    reg = LinearRegression().fit((invns).reshape(-1, 1), trend)
    return np.float64(reg.predict(np.array(INV_THRESH).reshape(-1, 1))[0])


@dataclass
class DelayPredictionTrend:
    """
    Represents a moving maximum delay prediction trend over the dataset.
    Not very useful. Implemented speculatively to disappointing results.
    """
    idxes: NDArray[np.int64]
    trend: NDArray[np.float64]

    @property
    def inv_idxes(self) -> NDArray[np.float64]:
        """
        The inverse maximum delay trend indexes.
        """
        return 1 / np.array(self.idxes, dtype=np.float64)


def maximal_delay_prediction_trend(
        ns: NDArray[np.int32],
        invns: NDArray[np.float64],
        trend: NDArray[np.float64],
        thres_samp=3,
        INV_THRESH: np.float64 = 0) -> DelayPredictionTrend:
    """
    Calculates the maximum delay prediction trend over the dataset.
    Not very useful. Implemented speculatively to disappointing results.
    :param ns: Array of maximum delay trend indices
    :param invns: Array of inverse maximum delay trend indices
    :param trend: Array of maximum delay trend values
    :param thres_samp: Sample size for maximum delay trend calculations (window size)
    :param INV_THRESH: Threshold y=T to calculate x-intercepts.
    """
    prediction_idxes: List[np.int32] = list()
    prediction_trend: List[np.float64] = list()

    if len(invns) < thres_samp + 2:
        return np.array(linear_regression_intercept(invns[1:], trend[1:], INV_THRESH=INV_THRESH))

    for i in range(thres_samp + 1, len(invns)):
        j = i - thres_samp
        window_invns = invns[j:i]
        window_trend = trend[j:i]
        prediction = linear_regression_intercept(window_invns, window_trend, INV_THRESH=INV_THRESH)
        prediction_idxes.append(ns[i])
        prediction_trend.append(prediction)

    return DelayPredictionTrend(np.array(prediction_idxes, dtype=np.int32),
                                np.array(prediction_trend, dtype=np.float64))


def estimate_global_critical_delay(
        max_delay_idxes: NDArray[np.int64],
        max_delay_trend: NDArray[np.float64],
        thres_samp: int = 3,
        INV_THRESH: np.float64 = 0
) -> np.float64:
    """
    Estimates the maximum delay of the circuit as indicated by the finite transient analysis (the maximum propagation
    delay the design will produce over infinite transitions)
    :param max_delay_idxes: Array of indices corresponding to maximum delay trend indexes
    :param max_delay_trend: Array of delays corresponding to maximum delay trend indexes
    :param thres_samp: Sample size for maximum delay trend calculations
    :param INV_THRESH: Threshold y=T to calculate x-intercepts.
    :return: x-intercept (indicating maximum delay)
    """
    last_trend = max_delay_trend[-thres_samp:]
    last_idxes = max_delay_idxes[-thres_samp:]
    last_inv_idxes = 1 / np.array(last_idxes, dtype=np.float64)
    prediction = linear_regression_intercept(last_inv_idxes, last_trend, INV_THRESH=INV_THRESH)
    return prediction

