import deeptrack as dt


def single_particle_model(input_shape, loss):
    return dt.models.AutoTracker(input_shape=input_shape, loss=loss)