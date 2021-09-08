import argparse
import os

import tensorflow as tf
import autotracker
import matplotlib.pyplot as plt
import glob
import numpy as np

import time

parser = argparse.ArgumentParser(
    description="Train a label-free single-particle tracker",
)
parser.add_argument("dataset", metavar="d", type=str)
parser.add_argument("models", metavar="d", type=str)
parser.add_argument("--maxdt", dest="maxdt", type=int, default=100)

extra_methods = [autotracker.radialcenter]


def main():
    args = parser.parse_args()

    frames = autotracker.load(args.dataset)[:10000]

    frames = (frames - np.min(frames, axis=(1, 2, 3), keepdims=True)) / np.ptp(
        frames, axis=(1, 2, 3), keepdims=True
    )
    plt.figure(figsize=(10, 10))
    all_models = glob.glob(args.models)
    for model_path in all_models:
        _, model_name = os.path.split(model_path)

        model = autotracker.load_single_particle_model(model_path)
        start = time.time()
        predictions = model.predict(frames, batch_size=32)
        eval_time = time.time() - start

        pred_msd = autotracker.msd(predictions, args.maxdt)
        plt.plot(pred_msd, linewidth=1.5)

        print(
            f"Evaluated {model_name} on {frames.shape[0]} images in \t {eval_time:.3f}s"
        )

        del model
        tf.keras.backend.clear_session()

    for comparison_method in extra_methods:
        start = time.time()
        predictions = comparison_method(frames)
        eval_time = time.time() - start
        pred_msd = autotracker.msd(predictions, args.maxdt)
        plt.plot(pred_msd, linewidth=1.5, linestyle=":")

        print(
            f"Evaluated {comparison_method.__name__} on {frames.shape[0]} images in \t {eval_time:.3f}s"
        )

    plt.xlabel("Delta t")
    plt.ylabel("MSD (px^2)")
    plt.legend(all_models + [f.__name__ for f in extra_methods])
    plt.show()


if __name__ == "__main__":
    main()