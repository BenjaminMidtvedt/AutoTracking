import deeptrack as dt
import random
import itertools

__all__ = ["dataloader"]


def dataloader(dataset):
    return dt.Value(lambda: random.choice(dataset)) >> dt.NormalizeMinMax()
