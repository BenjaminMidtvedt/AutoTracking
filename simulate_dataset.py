#%%
from datasets.sphere import pipeline
import numpy as np
import os
import pandas as pd
import tqdm

data_pipeline = pipeline(0)

data_size = 1000

out = np.zeros((data_size, 64, 64, 1))
df = pd.DataFrame(columns=["x", "y", "snr", "rotate"])

for i in tqdm.tqdm(range(data_size)):
    
    im = data_pipeline.update()()
    props = {
        "x": im.get_property("position")[1],
        "y": im.get_property("position")[0],
        "snr": im.get_property("snr"),
        "rotate":im.get_property("rotate") or 0
    }
    
    df = df.append(props, ignore_index=True)
    out[i] = im

os.makedirs("datasets/simulated_sphere/", exist_ok=True)
np.save("datasets/simulated_sphere/sphere", out)
df.to_csv("datasets/simulated_sphere/labels.csv")


#%%
