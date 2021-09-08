import deeptrack as dt
import numpy as np


def pipeline(args):

    image_size = 64

    optics = dt.Fluorescence(output_region=(0, 0, image_size, image_size))
    particle = dt.Sphere(
        position=lambda: image_size / 2 + np.random.randn(2) * 2, radius=5 * dt.units.px
    )
    noise = dt.Poisson(snr=lambda: 1 + np.random.rand() * 19, background=0.1)
    sample = optics(particle) + 0.1 >> noise
    return sample
