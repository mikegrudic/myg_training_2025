from numba import jit
import fitparse
import numpy as np
import datetime
import os
import pickle
from functools import cache
from pathlib import Path

CACHE_DIR = Path(".fitfile_cache")


@jit(fastmath=True, error_model="numpy")
def smooth(x, tau=60.0):
    """Implements a simple single-pole IIR low-pass filter with specified e-folding time tau"""
    x = x.astype(np.float64)
    if tau == 0.0:
        return x

    result = np.zeros(len(x))
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = x[i] * (1.0 / tau) + result[i - 1] * (1 - 1.0 / tau)
    return result


def _parse_fitfile_raw(f):
    """Parse a .fit file into raw numpy arrays (no smoothing or trimming)."""
    fitfile = fitparse.FitFile(f)
    data_values = {}
    data_units = {}
    for record in fitfile.get_messages("record"):
        for data in record:
            if data.name not in data_values:
                data_values[data.name] = []
                data_units[data.name] = data.units
            if data.value is None:
                data.value = np.nan
            data_values[data.name].append(data.value)
    for k, v in data_values.items():
        data_values[k] = np.array(v)
    return data_values, data_units


def _load_or_parse(f):
    """Return raw parsed arrays for a .fit file, using a pickle cache keyed by mtime."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / (Path(f).name + ".pkl")
    if cache_path.exists() and os.path.getmtime(cache_path) >= os.path.getmtime(f):
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)
    result = _parse_fitfile_raw(f)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    return result


@cache
def fitfile_to_data(f, smoothing_seconds=0.0, seconds_tocut=0):
    """Parses a fitparse file into two dicts containing the data fields and their units respectively"""
    data_values, data_units = _load_or_parse(f)
    out = {}
    for k, v in data_values.items():
        if smoothing_seconds:
            if not (
                isinstance(v[0], str) or isinstance(v[0], datetime.datetime) or k == "distance" or v.dtype == object
            ):
                v = smooth(v, tau=smoothing_seconds)
        out[k] = v[seconds_tocut:]
    return out, data_units
