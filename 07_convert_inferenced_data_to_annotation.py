#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools
import pickle

data_path = Path.cwd() / "data"
input_dir = data_path / "inference_data"
output_dir = data_path / "inference_filtered_data"
pickle_path = input_dir / "inference.pkl"
input_annotation_path = input_dir / "annotation.txt"

output_dir.mkdir(exist_ok=True)


# In[ ]:


with open(pickle_path, "rb") as f:
    load_data = pickle.load(f)
with open(input_annotation_path, "r") as f:
    load_annotations = f.readlines()

load_annotations = "".join(load_annotations)
#load_annotations = "#\n" + load_annotations

annotations = load_annotations.split("#")
annotations = annotations[1:]

annotations = [a.strip().split("\n")[0:2] for a in annotations]
annotations = [(a[0], a[1]) for a in annotations]

print("#inferenced data: ", len(load_data), ", #annotations: ", len(annotations))
assert(len(load_data) == len(annotations))

inferred_data = zip(annotations, load_data)
inferred_data = list(inferred_data)


# In[ ]:


confidence_threshold = 0.2

for (image_filename, image_size), areas in tqdm(inferred_data):
    filename = image_filename[:-4]
    output_txt_path = output_dir / (filename + ".txt")
    areas = areas[0]
    #print(filename)
    
    areas = [(int(x0), int(y0), int(x1), int(y1)) for (x0, y0, x1, y1, p) in areas if p > confidence_threshold]
    
    if (len(areas) <= 0):
        continue

    annotation_text = [" ".join([str(d) for d in l]) + " 1" for l in areas]
    annotation_text = "\n".join(annotation_text)
    annotation_text = f"{image.shape[1]} {image.shape[0]}\n{len(areas)}\n{annotation_text}"
    
    with open(output_txt_path, "w") as f:
        f.write(annotation_text)

print("output done.")


# In[ ]:


# import matplotlib.pyplot as plt
# plt.hist([p for p in probs if p >= 0.1], bins=32)


# In[ ]:




