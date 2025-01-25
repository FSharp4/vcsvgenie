from io import StringIO
from os import linesep
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame, read_csv

TITLE_HEADER_IDX = 1
BEGIN_DATA_IDX = 6

def read_vcsv(path: Path) -> Tuple[DataFrame, List[str]]:
    lines = path.read_text().splitlines()
    titles = lines[TITLE_HEADER_IDX][1:].split(",;")
    series = []
    for title in titles:
        series.append(f"{title} X")
        series.append(f"{title} Y")
    dataframe = read_csv(StringIO(linesep.join(lines[BEGIN_DATA_IDX:])))
    return dataframe, titles

def read_vcsv_as_numpy(path: Path) -> Tuple[NDArray[np.float64], List[str]]:
    df, titles = read_vcsv(path)
    arr = df.to_numpy().astype(np.float64)
    return arr, titles