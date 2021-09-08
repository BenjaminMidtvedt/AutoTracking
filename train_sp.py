import argparse
import autotracker
import matplotlib.pyplot as plt
import deeptrack as dt
import numpy as np

parser = argparse.ArgumentParser(
    description="Train a label-free single-particle tracker",
)
parser.add_argument("filename", metavar="d", type=str)
parser.add_argument("--batch_size", dest="batch_size", type=int, default=8)
parser.add_argument("--epochs", dest="epochs", type=int, default=20)
parser.add_argument("--trainframes", dest="train_frames", type=str, default=":")
parser.add_argument("--lossfn", dest="lossfn", type=str, default="mae")
parser.add_argument("--prefix", dest="prefix", type=str, default="")
parser.add_argument("--radius", dest="radius", type=int, default=2)
parser.add_argument("--rotate", dest="rotate", type=int, default=1)
parser.add_argument("--sigma", dest="sigma", type=int, default=0.01)


def main():
    args = parser.parse_args()

    frames = autotracker.load(args.filename)

    training_set = eval(f"frames[{args.train_frames}]")

    dataloader = autotracker.dataloader(training_set) >> dt.Gaussian(sigma=args.sigma)
    model = autotracker.single_particle_model(
        input_shape=frames.shape[1:], loss=args.lossfn
    )

    # generator = model.data_generator(dataloader, batch_size=args.batch_size)
    # with generator:

    model.fit(
        dataloader,
        epochs=args.epochs,
        batch_size=args.batch_size,
        generator_kwargs={"radius": args.radius, "rotate": args.rotate},
    )

    autotracker.save(model, args)


if __name__ == "__main__":
    main()