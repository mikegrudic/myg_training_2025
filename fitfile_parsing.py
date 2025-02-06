from numba import jit
import fitparse
import numpy as np
import datetime


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


def fitfile_to_data(f, smoothing_seconds=0.0, seconds_tocut=0):
    """Parses a fitparse file into two dicts containing the data fields and their units respectively"""
    fitfile = fitparse.FitFile(f)

    data_values = {}
    data_units = {}
    for record in fitfile.get_messages("record"):
        for data in record:
            if data.name not in data_values.keys():
                data_values[data.name] = []
                data_units[data.name] = data.units

            if data.value is None:
                data.value = np.nan

            data_values[data.name].append(data.value)

    for k, v in data_values.items():
        v = np.array(v)
        if smoothing_seconds:
            if not (isinstance(v[0], str) or isinstance(v[0], datetime.datetime) or k == "distance"):
                v = smooth(v, tau=smoothing_seconds)
        data_values[k] = v[seconds_tocut:]
    return data_values, data_units
