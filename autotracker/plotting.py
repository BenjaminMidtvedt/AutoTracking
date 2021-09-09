import numpy as np
import matplotlib.pyplot as plt

def binned_error(x, y, nbins, **kwargs):
    vmin = np.min(x)
    vmax = np.max(x)
    bin_edges = np.linspace(vmin, vmax, nbins + 1)

    _x = x
    _y = y
    
    y = np.array([np.mean( _y[(_x >= bin_edges[i]) & (_x < bin_edges[i+1]) ], axis=0) for i in range(nbins)])
    x = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    plt.subplot(1, 3, 2)
    plt.plot(x, y, **kwargs)
    plt.yscale("log")

