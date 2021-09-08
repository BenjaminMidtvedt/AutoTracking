import numpy as np
import deeptrack as dt
from deeptrack.extras.radialcenter import radialcenter as _radialcenter
import scipy.ndimage


def msd(positions, max_dt=100):
    out = np.zeros((max_dt))
    for delta in range(1, max_dt + 1):
        out[delta - 1] = np.square(positions[delta:] - positions[:-delta]).mean()

    dx = out[1] - out[0]
    d0 = out[0] - dx
    out = out - d0
    return out


def radialcenter(dataset):
    dataset = np.mean(dataset, axis=-1)
    return np.array([_radialcenter(d) for d in dataset])


def centroid(dataset):
    dataset = np.mean(dataset, axis=-1)
    dataset = dataset - np.mean(dataset, axis=(1, 2), keepdims=True)

    return np.array([scipy.ndimage.center_of_mass(dataset) for d in dataset])