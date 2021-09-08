__all__ = ["load", "save", "load_single_particle_model", "load_multi_particle_model"]

import os
import numpy as np
import tensorflow as tf
import deeptrack as dt
from tqdm import tqdm


def load(filename: str):
    *_, ext = filename.split(os.path.extsep)

    if ext == "py":
        return load_python(filename)

    return load_video(filename)


def load_python(filename):
    pass


def load_video(filename):
    import cv2

    cap = cv2.VideoCapture(filename)

    n_frames, width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    frames = np.empty((n_frames, width, height, 3))

    print("Reading video...")
    for t in tqdm(range(n_frames)):
        _, frames[t] = cap.read()

    return frames


def load_image(filename):
    pass


def save(model, args):

    _out_path = f"checkpoints/{args.prefix}{os.path.split(args.filename)[-1]}"
    out_path = _out_path
    idx = 0
    while os.path.exists(out_path):
        idx += 1
        out_path = f"{_out_path}_{idx}"
    out_path = os.path.normcase(out_path)
    os.makedirs(out_path, exist_ok=True)

    model.save(out_path)


def load_single_particle_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "AutoTrackerModel": dt.models.autotrack.AutoTracker.AutoTrackerModel
        },
    )


def load_multi_particle_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "AutoTrackerModel": dt.models.autotrack.AutoTracker.AutoTrackerModel
        },
    )
