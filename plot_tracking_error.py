import argparse
import os

import tensorflow as tf
import autotracker
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd

import time

pd.DataFrame.to_numpy

parser = argparse.ArgumentParser(
    description="Train a label-free single-particle tracker",
)
parser.add_argument("dataset", metavar="d", type=str)
parser.add_argument("models", metavar="m", type=str)
parser.add_argument("--maxdt", dest="maxdt", type=int, default=100)

extra_methods = [autotracker.radialcenter]


def main():
    args = parser.parse_args()

    frames, labels = autotracker.load(args.dataset)

    frames = (frames - np.min(frames, axis=(1, 2, 3), keepdims=True)) / np.ptp(
        frames, axis=(1, 2, 3), keepdims=True
    )
    plt.figure(figsize=(10, 10))
    all_models = glob.glob(args.models)
    for model_path in all_models:
        _, model_name = os.path.split(model_path)

        model = autotracker.load_single_particle_model(model_path) 
        start = time.time()
        predictions = model.predict(frames, batch_size=32) + np.array(frames.shape[1:3])/2
        eval_time = time.time() - start

        print(
            f"Evaluated {model_path} on {frames.shape[0]} images in \t {eval_time:.3f}s"
        )

        x_err = labels["x"].to_numpy() - predictions[:, 1]
        x_err = np.abs(x_err - np.mean(x_err))
        y_err = np.abs(labels["y"].to_numpy() - predictions[:, 0])
        y_err = np.abs(y_err - np.mean(y_err))

        error = (x_err + y_err) / 2

        variable = labels["snr"].to_numpy()


        autotracker.binned_error(variable, error, 10)
        
        

        for comparison_method in extra_methods:
            start = time.time()
            predictions = comparison_method(frames)
            eval_time = time.time() - start
            

            print(
                f"Evaluated {comparison_method.__name__} on {frames.shape[0]} images in \t {eval_time:.3f}s"
            )

            error = (
                np.abs(labels["x"].to_numpy() - predictions[:, 0]) + 
                np.abs(labels["y"].to_numpy() - predictions[:, 1])
            ) / 2
            autotracker.binned_error(variable, error, 10, c="gray")

        plt.xlabel("SNR")
        plt.ylabel("Absolute error (px)")
        plt.legend(all_models + [f.__name__ for f in extra_methods])
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{model_name}_tracking_error.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()