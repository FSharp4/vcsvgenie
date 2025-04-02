# VCSV Genie

This repo details methods for parsing transient simulation VCSV files from Cadence Virtuoso.

## Installation

Install via `pip`:
```bash
pip install vcsvgenie
```

> https://pypi.org/project/vcsvgenie/

## Installation from source (using Poetry)

1. Clone this repo

```bash
git clone https://github.com/FSharp4/vcsvgenie.git
cd vcsvgenie
```

2. Install to an environment

```bash
# If you haven't installed poetry:
pip install poetry

# Install project + dependencies to virtual environment
poetry install
```

3. Copy over a vcsv file of your choice, and use `vcsvgenie` to process the file, extracting propagation delays, printing traces, etc.

## Installation from source

1. Clone this repo
```bash
git clone https://github.com/FSharp4/vcsvgenie.git
cd vcsvgenie
```

2. Build this repo
```bash
pip install build
python -m build
```

3. Install the repo
```bash
pip install -e .
```

4. Copy over a vcsv file of your choice, and use `vcsvgenie` to process the file, extracting propagation delays, printing traces, etc.

Example files can be seen in `vcsvgenie/_dev`.

**Remark**: In order for VCSV genie to automatically parse your file for propagations, you need to specify which signals are 'outputs' and which signals are inputs. 
- Signals must be 'outputs' for propagation delay calculations to be performed on them with respect to inputs.
- Within Virtuoso Schematics/Layouts, these signals may be inputOutput or intermediate.

To specify which signals are inputs/outputs, supply lists of signal names to the `input` and `output` constructor arguments for `TransientResultSpecification`. (See Usage for an example).)

## Functionality

### Transient Analysis

Users can:
- Create pandas dataframes or numpy arrays from VCSV files
- Create $(x, y)$ dataseries of individual waveforms (note that all timestamp $x$ vectors are the same in a VCSV file, and are not uniformly spaced)
- Create collections of waveforms from specifications (`vcsvgenie.transient_waveform.TransientResult`) 
- Recognize signal buses using caret notation (i.e., A<3:0>) by specifying the individual signals in the `TransientResultSpecification`
- Digitize and tabulate signal bus data
- Identify and measure critical path(s) propagation delay(s) via `vcsvgenie.transient_waveform.critical_propagation_delays` and `quasicritical_propagation_delays`
- Estimate global worst-case delays via `vcsvgenie.transient_waveform.estimate_global_critical_delay`

### DC Analysis

Users can:
- Load waveforms from a DC analysis into a rudimentary DC Results object (no functionality yet)
- Measure Read SRAM Noise Margin
- Measure Write SRAM Noise Margin

## Usage

**General Example**:

```python
from pathlib import Path
from pprint import pprint

from vcsvgenie.read import read_vcsv
from vcsvgenie.transient_waveform import TransientResultSpecification, average_propagation_delays_by_category,
    maximum_propagation_delays_by_category
from vcsvgenie.waveform import construct_waveforms

path = Path("example.vcsv")
dataframe, titles = read_vcsv(path)
waveforms = construct_waveforms(dataframe, titles)

specification = TransientResultSpecification(
    inputs=[
        '/A<3>', '/A<2>', '/A<1>', '/A<0', '/B<3>', '/B<2>', '/B<1>', '/B<0>', 'Clk'
    ],
    outputs=['/z<7>', '/z<6>', '/z<5>', '/z<4>', '/z<3>', '/z<2>', '/z<1>', '/z<0>'],
    clock_period=1e-9,
    logic_threshold=0.5  # volts
)

results = specification.interpret(waveforms)
results.find_transitions()
results.find_propagations()

averages = average_propagation_delays_by_category(results.propagations)
# pprint(results.propagations)
# pprint(averages)

maxima = maximum_propagation_delays_by_category(results.propagations)
# pprint(maxima)

results.digitize()
results.resolve_buses()
bus_data = results.tabulate_bus_data()
bus_data.to_csv("bus_data.csv")

results.plot(separate=True)
```

**Retrieving observed worst-case delays**:

```python
from pprint import pprint

n = 10 # number of delays to retrieve
quasicritical_delays = transient_waveform.quasicritical_propagation_delays(results.propagations, n)
pprint([str(quasicritical) for quasicritical in quasicritical_delays])
```

**Estimating global worst-case delays (in simulations on random inputs)**:

```python
maxtpd_idx, maxtpd = transient_waveform.find_max_delay_trend(results.delays)
transient_waveform.plot_max_delay_trend(maxtpd_idx, maxtpd, show=True)

global_estimation = transient_waveform.estimate_global_critical_delay(maxtpd_idx, maxtpd, thres_samp=thres_samp)
```

## Dependencies

- numpy
- matplotlib
- pandas
- sortedcontainers
- (optional): jupyter

## Contribution

This code targets python 3.11 syntax.

This codebase follows semver guidelines:
- Patch versions indicate bug fixes or documentation adjustments with no intentional functionality changes
- Minor versions indicate functionality changes or additions with no breaking changes
- Major versions indicate breaking functionality changes
