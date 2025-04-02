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
    """
    Reads a VCSV file at the path into a read_dataframe, extracting the list of read_titles separately.
    :param path: Path to the VCSV file
    :return: DataFrame, list of read_titles
    """
    lines = path.read_text().splitlines()
    titles = lines[TITLE_HEADER_IDX][1:].split(",;")
    series = []
    for title in titles:
        series.append(f"{title} X")
        series.append(f"{title} Y")
    # dataframe = read_csv(StringIO(linesep.join(lines[BEGIN_DATA_IDX:])), names=titles)
    lines.insert(BEGIN_DATA_IDX, lines[BEGIN_DATA_IDX])
    dataframe = read_csv(StringIO(linesep.join(lines[BEGIN_DATA_IDX:])))
    return dataframe, titles

def read_vcsv_as_numpy(path: Path) -> Tuple[NDArray[np.float64], List[str]]:
    """
    Reads a VCSV file at the path into a numpy array, extracting the list of read_titles separately.
    :param path: Path to the VCSV file
    :return: Two-dimensional NDArray[np.float64] (number of points x (2 * number of read_titles)), list of read_titles
    """
    df, titles = read_vcsv(path)
    arr = df.to_numpy().astype(np.float64)
    return arr, titles