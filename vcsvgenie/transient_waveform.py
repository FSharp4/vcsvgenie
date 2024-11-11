from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
from warnings import warn
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


@dataclass
class WaveForm:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    title: str

@dataclass
class Transition:
    TransitionType = Literal['Rising', 'Falling']
    transition_type: TransitionType
    series: str
    series_type: Literal['Input', 'Output']
    interval: int

@dataclass
class Propagation:
    source: str
    destination: str
    departure: float
    arrival: float
    propagation_type: Literal['Rising', 'Falling']
    delay: float
    interval: int

    def type_label(self) -> str:
        return f"{self.source} -> {self.destination}"

class TransientResult:
    def __init__(self, inputs: List[WaveForm], outputs: List[WaveForm], name: str = "Transient Results"):
        self.timestamps = inputs[0].x
        self.inputs: Dict[str, NDArray[np.float64]] = dict()
        self.outputs: Dict[str, NDArray[np.float64]] = dict()
        for input in inputs:
            if not np.allclose(input.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.inputs[input.title] = input.y
        
        for output in outputs:
            if not np.allclose(output.x, self.timestamps):
                raise ArithmeticError("Waveform Timestamps are not aligned")

            self.outputs[output.title] = output.y

        self.name = name
        self.timestep = 1e-9
        self.eps_n_timestamps = 10
        self._transitions: List[List[Transition]] = list()
        self._interval_start_idxs: NDArray[np.int32] = np.zeros(1, dtype=np.int32)
        self._interval_end_idxs: NDArray[np.int32] = np.zeros(1, dtype=np.int32)
        self._propagations: List[Propagation] = list()

    @property
    def propagations(self) -> List[Propagation]:
        if len(self._propagations) == 0:
            raise Exception("Propagations not calculated (call TransientResult.find_propagations first)")
        
        return self._propagations
    
    @property
    def transitions(self) -> List[List[Transition]]:
        if len(self._transitions) == 0:
            raise Exception("Transitions not calculated (call TransientResult.find_transitions first)")
    
        return self._transitions

    @property
    def interval_start_idxs(self) -> NDArray[np.int32]:
        if np.all(self._interval_start_idxs == 0):
            raise Exception("Interval Start Timestamp Indexes not calculated (call TransientResult.find_transitions first)")
        
        return self._interval_start_idxs

    @property
    def interval_end_idxs(self) -> NDArray[np.int32]:
        if np.all(self._interval_end_idxs == 0):
            raise Exception("Interval End Timestamp Indexes not calculated (call TransientResult.find_transitions first)")
        
        return self._interval_end_idxs
    
    def plot(self, save: bool = False, display: bool = True, separate: bool = False) -> None:
        if not display and not save:
            warn("TransientResult.plot called without save nor display; defaulting to display")
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
            plt.grid(visible = True, which = 'both', axis = 'both')
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
                plt.grid(visible=True, which='both', axis='both')
                if display:
                    plt.show()
                else:
                    plt.savefig(f"TransientResult__plot {self.name}_{title.replace('/', '_')}.png")
                    plt.close()

            for title in self.outputs:
                plt.figure()
                plt.plot(self.timestamps, self.outputs[title], 'r')
                plt.xlabel("Timestamp (s)")
                plt.ylabel("Voltage (V)")
                plt.title(f"{self.name} {title}")
                plt.grid(visible=True, which='both', axis='both')
                if display:
                    plt.show()
                else:
                    plt.savefig(f"TransientResult__plot {self.name}_{title.replace('/', '_')}.png")
                    plt.close()

    def find_transitions(self, timestep: float = 1e-9, eps_n_timestamps: int = 1, LOGIC_THRESHOLD: float = 0.5) -> None:
        self.timestep = timestep
        self.eps_n_timestamps = eps_n_timestamps
        ending_timestamp = self.timestamps[-1]
        n_iterations = np.int32((ending_timestamp + timestep/2) // timestep) - 1

        start_timestamp_idxs = np.zeros(n_iterations, dtype=np.int32)
        end_timestamp_idxs = np.zeros(n_iterations, dtype=np.int32)
        transitions: List[List[Transition]] = list()
        for n in range(n_iterations):
            start = timestep * n
            end = timestep * (n + 1)
            start_timestamp_idxs[n] = np.searchsorted(self.timestamps, start, side='right')
            end_timestamp_idxs[n] = np.searchsorted(self.timestamps, end, side='right') - eps_n_timestamps
            transitions.append([])

        for input_series in self.inputs.keys():
            data = self.inputs[input_series]
            starts_data = data[start_timestamp_idxs] > LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > LOGIC_THRESHOLD
            
            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    transition_type = 'Rising' if ends_data[idx] else 'Falling'
                    transition = Transition(transition_type, input_series, 'Input', idx)
                    transitions[idx].append(transition)
        
        for output_series in self.outputs.keys():
            data = self.outputs[output_series]
            starts_data = data[start_timestamp_idxs] > LOGIC_THRESHOLD
            ends_data = data[end_timestamp_idxs] > LOGIC_THRESHOLD
            
            check_transitioned = ends_data != starts_data
            for idx, did_transition in enumerate(check_transitioned):
                if did_transition:
                    transition_type = 'Rising' if ends_data[idx] else 'Falling'
                    transition = Transition(transition_type, output_series, 'Output', idx)
                    transitions[idx].append(transition)
        
        self._transitions = transitions
        self._interval_start_idxs = start_timestamp_idxs
        self._interval_end_idxs = end_timestamp_idxs

    def find_propagations(self, LOGIC_THRESHOLD: float = 0.5) -> None:
        propagations: List[Propagation] = list()
        for idx, timestep_transitions in enumerate(self.transitions):
            if len(timestep_transitions) < 2:
                continue

            input_transitions: List[Transition] = [transition for transition in timestep_transitions if transition.series_type == 'Input']
            output_transitions: List[Transition] = [transition for transition in timestep_transitions if transition.series_type == 'Output']
            if len(output_transitions) == 0:
                continue

            for input_transition in input_transitions:
                input_transition_timestamp = self.interpolate_transition_timestamp(LOGIC_THRESHOLD, input_transition)
                for output_transition in output_transitions:
                    output_transition_timestamp = self.interpolate_transition_timestamp(LOGIC_THRESHOLD, output_transition)
                    propagation_delay = output_transition_timestamp - input_transition_timestamp
                    propagations.append(
                        Propagation(
                            input_transition.series, 
                            output_transition.series, 
                            input_transition_timestamp, 
                            output_transition_timestamp, 
                            output_transition.transition_type, 
                            propagation_delay, 
                            input_transition.interval
                        )
                    )
        
        self._propagations = propagations
        # return propagations

    def interpolate_transition_timestamp(self, LOGIC_THRESHOLD: float, transition: Transition) -> float:
        sweep_start_timestamp: np.int32 = self.interval_start_idxs[transition.interval]
        sweep_end_timestamp: np.int32 = self.interval_end_idxs[transition.interval] + 1
        data_interval: NDArray[np.float64]
        if transition.series_type == 'Input':
            data_interval = self.inputs[transition.series][sweep_start_timestamp:sweep_end_timestamp]
        else:
            data_interval = self.outputs[transition.series][sweep_start_timestamp:sweep_end_timestamp]

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


def find_transitions(transient_result: TransientResult, timestep: float = 1e-9, eps_n_timestamps: int = 10) -> List[List[Transition]]:
    transient_result.find_transitions(timestep=timestep, eps_n_timestamps=eps_n_timestamps)[0]
    return transient_result.transitions

        
def construct_waveforms(waveform_dataframe: DataFrame, titles: List[str]) -> List[WaveForm]:
    waveforms: List[WaveForm] = list()
    for idx, title in enumerate(titles):
        x, y = waveform_dataframe.iloc[:, 2 * idx].to_numpy(dtype=np.float64), waveform_dataframe.iloc[:, 2 * idx + 1].to_numpy(dtype=np.float64)
        waveform = WaveForm(x, y, title)
        waveforms.append(waveform)
    return waveforms

@dataclass
class TransientResultSpecification:
    inputs: List[str]
    outputs: List[str]

    def interpret(self, waveforms: List[WaveForm], name: str = "Transient Results"):
        input_waveforms = []
        output_waveforms = []
        for waveform in waveforms:
            if waveform.title in self.inputs:
                input_waveforms.append(waveform)
            elif waveform.title in self.outputs:
                output_waveforms.append(waveform)
            
        return TransientResult(input_waveforms, output_waveforms, name=name)

def average_propagation_delays_by_category(propagations: List[Propagation]) -> Dict[str, Tuple[float, float]]:
    propagation_dictionary: Dict[str, Tuple[List[Propagation], List[Propagation]]] = dict()
    averages: Dict[str, Tuple[float, float]] = dict()
    for propagation in propagations:
        label = propagation.type_label()
        if label not in propagation_dictionary.keys():
            propagation_dictionary[label] = ([], [])

        if propagation.propagation_type == 'Rising':
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