import deeptrack as dt


def single_particle_model(input_shape, loss):
    model = dt.models.Convolutional(
            input_shape=input_shape,
            conv_layers_dimensions=[32, 64, 128],
            dense_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=2,
        )
    return dt.models.AutoTracker(model, input_shape=input_shape, loss=loss)