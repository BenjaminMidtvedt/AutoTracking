import deeptrack as dt
import numpy as np


def pipeline(args):

    image_size = 64

    optics = dt.Fluorescence(
        output_region=(0, 0, image_size, image_size),
        NA=0.8,                
        resolution=1e-6,     
        magnification=10,
        wavelength=680e-9
    )
    particle = dt.Ellipse(
        position=lambda: image_size / 2 + np.random.randn(2) * 2, radius=1 * dt.units.um
    )
    noise = dt.Poisson(snr=lambda: 1 + np.random.rand() * 19, background=0.2)
    sample = optics(particle) + 0.2 >> noise
    return sample
