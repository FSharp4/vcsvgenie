# VCSV Genie

This repo details methods for parsing transient simulation VCSV files from Cadence Virtuoso.

## Installation

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

VCSV Genie is presently limited to processing VCSV files produced from transient simulaitons in Cadence Virtuoso. Users can:
- Create pandas dataframes from VCSV files
- Create $(x, y)$ dataseries of individual waveforms (note that all timestamp $x$ vectors are the same in a VCSV file, and are not uniformly spaced)
- Create collections of waveforms from specifications (`vcsvgenie.transient_waveform.TransientResult`) 

## Usage

```python
from pathlib import Path
from pprint import pprint

from vcsvgenie.read import read_vcsv
from vcsvgenie.transient_waveform import TransientResultSpecification, average_propagation_delays_by_category, construct_waveforms

path = Path("example.vcsv")
dataframe, titles = read_vcsv(path)
waveforms = construct_waveforms(dataframe, titles)

specification = TransientResultSpecification(
    inputs = [
        '/D', '/Clk', '/Reset'
    ],
    outputs = ['/Q']
)

results = specification.interpret(waveforms)
results.find_transitions()
results.find_propagations()
averages = average_propagation_delays_by_category(results.propagations)
pprint(results.propagations)
pprint(averages)

results.plot(separate=True)
```

## Dependencies

- numpy
- matplotlib
- pandas
- (optional): jupyter