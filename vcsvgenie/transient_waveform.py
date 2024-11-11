from dataclasses import dataclass
from typing import Dict, List
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
                plt.xlabel("Timestamp (ns)")
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
                plt.plot(self.timestamps, self.inputs[title], 'r')
                plt.xlabel("Timestamp (ns)")
                plt.ylabel("Voltage (V)")
                plt.title(f"{self.name} {title}")
                plt.grid(visible=True, which='both', axis='both')
                if display:
                    plt.show()
                else:
                    plt.savefig(f"TransientResult__plot {self.name}_{title.replace('/', '_')}.png")
                    plt.close()
        
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
