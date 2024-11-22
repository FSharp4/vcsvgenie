from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
from warnings import warn
from matplotlib import pyplot as plt
from sortedcontainers import SortedDict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame


@dataclass
class Bus:
    title: str
    members: List[str]
    powers: List[int]

    def equals(self, bus: "Bus"):
        if self.title != bus.title:
            return False
        if self.members != bus.members:
            return False
        if self.powers != bus.powers:
            return False

        return True


@dataclass
class WaveForm:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    title: str


@dataclass
class Transition:
    transition_type: Literal["Rising", "Falling"]
    series: str
    series_type: Literal["Input", "Output"]
    interval: int

    def __str__(self) -> str:
        return f"i={self.interval} [{self.series_type}] {self.series} {self.transition_type} transition observed"


@dataclass
class Propagation:
    source: str
    destination: str
    departure: float
    arrival: float
    propagation_type: Literal["Rising", "Falling"]
    delay: float
    interval: int

    def type_label(self) -> str:
        return f"{self.source} -> {self.destination}"
    
    def __str__(self) -> str:
        return f"[i={self.interval}] {self.source} -> {self.destination}: {self.delay}"


class TransientResult:
    def __init__(
        self,
        inputs: List[WaveForm],
        outputs: List[WaveForm],
        name: str = "Transient Results",
        input_bus_dict: Dict[str, Bus] = {},
        output_bus_dict: Dict[str, Bus] = {},
        clock_period: float = 1e-9,
        logic_threshold: float = 0.5,
        absolute_bus_bits: bool = True,
        bus_display_radix: Literal["Unsigned Decimal"] = "Unsigned Decimal",
    ):
        self.timestamps = inputs[0].x
        self.inputs: Dict[str, NDArray[np.float64]] = dict()
        self.outputs: Dict[str, NDArray[np.float64]] = dict()
        self.input_bus_spec = input_bus_dict
        self.output_bus_spec = output_bus_dict
        self.absolute_bus_bits = absolute_bus_bits
        self.bus_display_radix = bus_display_radix
        for input in inputs:
            if not np.allclose(input.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.inputs[input.title] = input.y

        for output in outputs:
            if not np.allclose(output.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.outputs[output.title] = output.y

        self.name = name
        self.clock_period = clock_period
        self.eps_n_timestamps = 10
        self.LOGIC_THRESHOLD = logic_threshold
        self._transitions: List[List[Transition]] = list()
        self._interval_start_idxs: NDArray[np.int32] = np.zeros(1, dtype=np.int32)
        self._interval_end_idxs: NDArray[np.int32] = np.zeros(1, dtype=np.int32)
        self._propagations: List[Propagation] = list()

        self.n_intervals = int(
            np.ceil((self.timestamps[-1] - self.timestamps[0]) / self.clock_period)
        )
        self.digital_inputs: Dict[str, NDArray[np.bool_]] = dict()
        self.digital_outputs: Dict[str, NDArray[np.bool_]] = dict()
        self.input_bus_values: Dict[str, NDArray[np.int32]] = dict()
        self.output_bus_values: Dict[str, NDArray[np.int32]] = dict()

    @property
    def propagations(self) -> List[Propagation]:
        if len(self._propagations) == 0:
            raise Exception(
                "Propagations not calculated (call TransientResult.find_propagations first)"
            )

        return self._propagations

    @property
    def transitions(self) -> List[List[Transition]]:
        if len(self._transitions) == 0:
            raise Exception(
                "Transitions not calculated (call TransientResult.find_transitions first)"
            )

        return self._transitions

    @property
    def interval_start_idxs(self) -> NDArray[np.int32]:
        if np.all(self._interval_start_idxs == 0):
            raise Exception(
                "Interval Start Timestamp Indexes not calculated (call TransientResult.find_transitions first)"
            )

        return self._interval_start_idxs

    @property
    def interval_end_idxs(self) -> NDArray[np.int32]:
        if np.all(self._interval_end_idxs == 0):
            raise Exception(
                "Interval End Timestamp Indexes not calculated (call TransientResult.find_transitions first)"
            )

        return self._interval_end_idxs

    def plot(
        self, save: bool = False, display: bool = True, separate: bool = False
    ) -> None:
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

    def find_transitions(self, eps_n_timestamps: int = 1) -> None:
        self.eps_n_timestamps = eps_n_timestamps
        ending_timestamp = self.timestamps[-1]

        start_timestamp_idxs = np.zeros(self.n_intervals, dtype=np.int32)
        end_timestamp_idxs = np.zeros(self.n_intervals, dtype=np.int32)
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

        for input_series in self.inputs.keys():
            data = self.inputs[input_series]
            starts_data = data[start_timestamp_idxs] > self.LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > self.LOGIC_THRESHOLD

            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    transition_type = "Rising" if ends_data[idx] else "Falling"
                    transition = Transition(transition_type, input_series, "Input", idx)
                    transitions[idx].append(transition)

        for output_series in self.outputs.keys():
            data = self.outputs[output_series]
            starts_data = data[start_timestamp_idxs] > self.LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > self.LOGIC_THRESHOLD

            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    transition_type = "Rising" if ends_data[idx] else "Falling"
                    transition = Transition(
                        transition_type, output_series, "Output", idx
                    )
                    transitions[idx].append(transition)

        self._transitions = transitions
        self._interval_start_idxs = start_timestamp_idxs
        self._interval_end_idxs = end_timestamp_idxs

    def find_propagations(self) -> None:
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
        for bus_name in self.input_bus_spec:
            input_bus = self.input_bus_spec[bus_name]
            signals = input_bus.members
            powers = input_bus.powers
            if self.absolute_bus_bits == False:
                powers = list(range(len(powers) - 1, -1, -1))
            
            self.input_bus_values[bus_name] = np.zeros((self.n_intervals), dtype=np.int32)
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
            
            self.output_bus_values[bus_name] = np.zeros((self.n_intervals), dtype=np.int32)
            for interval in range(self.n_intervals):
                value: int = 0
                for idx, signal in enumerate(signals):
                    value += self.digital_outputs[signal][interval] * (1 << powers[idx])
                
                self.output_bus_values[bus_name][interval] = value

    def tabulate_bus_data(self):
        input_table = DataFrame.from_dict(self.input_bus_values)
        output_table = DataFrame.from_dict(self.output_bus_values)
        table = pd.concat([input_table, output_table], axis=1)
        return table

        

    def interpolate_transition_timestamp(
        self, LOGIC_THRESHOLD: float, transition: Transition
    ) -> float:
        sweep_start_timestamp: np.int32 = self.interval_start_idxs[transition.interval]
        sweep_end_timestamp: np.int32 = self.interval_end_idxs[transition.interval] + 1
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


def find_transitions(
    transient_result: TransientResult,
    timestep: float = 1e-9,
    eps_n_timestamps: int = 10,
) -> List[List[Transition]]:
    transient_result.find_transitions(
        timestep=timestep, eps_n_timestamps=eps_n_timestamps
    )[0]
    return transient_result.transitions


def construct_waveforms(
    waveform_dataframe: DataFrame, titles: List[str]
) -> List[WaveForm]:
    waveforms: List[WaveForm] = list()
    for idx, title in enumerate(titles):
        x, y = waveform_dataframe.iloc[:, 2 * idx].to_numpy(
            dtype=np.float64
        ), waveform_dataframe.iloc[:, 2 * idx + 1].to_numpy(dtype=np.float64)
        waveform = WaveForm(x, y, title)
        waveforms.append(waveform)
    return waveforms


class BusValidationException(Exception):
    def __init__(self, bus_title: str, signal_title: str, *args):
        super.__init__(*args)
        self.bus_title = bus_title
        self.signal_title = signal_title


# @dataclass
class TransientResultSpecification:

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        input_buses: Dict[str, Bus] = {},
        output_buses: Dict[str, Bus] = {},
        logic_threshold: float = 0.5,
        clock_period: float = 1e-9,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.input_buses = input_buses
        self.output_buses = output_buses
        self.logic_threshold = logic_threshold
        self.clock_period = clock_period

    def infer_buses(self) -> None:
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
