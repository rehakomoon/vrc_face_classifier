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
import math

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_dir = data_path / "inference_data"
output_dir = data_path / "inference_data"
pickle_path = input_dir / "inference.pkl"
input_annotation_path = input_dir / "annotation.txt"

output_dir.mkdir(exist_ok=True)


# In[ ]:


with open(pickle_path, "rb") as f:
    load_data = pickle.load(f)
with open(input_annotation_path, "r") as f:
    load_annotations = f.readlines()

load_data = [l[0] for l in load_data]

load_annotations = "".join(load_annotations)
#load_annotations = "#\n" + load_annotations

annotations = load_annotations.split("#")
annotations = annotations[1:]

annotations = [a.strip().split("\n")[0:2] for a in annotations]

print("#inferenced data: ", len(load_data), ", #annotations: ", len(annotations))
assert(len(load_data) == len(annotations))

inferred_data = zip(annotations, load_data)
inferred_data = list(inferred_data)


# In[ ]:


confidence_threshold = 0.2
scale_factor = 0.1
scale_rep_num = 3

for (image_filename, image_size), areas in tqdm(inferred_data):
    filename = image_filename[:-4]
    image_w, image_h = image_size.split(" ")
    image_w = int(image_w)
    image_h = int(image_h)
    
    output_txt_path = output_dir / (filename + ".txt")
    
    areas = [(int(x0), int(y0), int(x1), int(y1)) for (x0, y0, x1, y1, p) in areas if p > confidence_threshold]
    
    areas_scaled = []
    
    for x0, y0, x1, y1 in areas:
        #print(x0, y0, x1, y1)
        orig_w, orig_h = (x1 - x0, y1 - y0)
        assert(orig_h > 0 and orig_w > 0)
        
        extend_ratio = (1.0 / (1.0 - scale_factor)) ** scale_rep_num - 1.0
        extend_w = math.floor(orig_w * extend_ratio * 0.5)
        extend_h = math.floor(orig_h * extend_ratio * 0.5)
        x0 = max(x0 - extend_w, 0)
        y0 = max(y0 - extend_h, 0)
        x1 = min(x1 + extend_w, image_w - 1)
        y1 = min(y1 + extend_h, image_h - 1)
        #print(x0, y0, x1, y1)
        assert(x0 < x1 and y0 < y1)

        areas_scaled.append((x0, y0, x1, y1))
    
    areas = areas_scaled
    
    if (len(areas) <= 0):
        continue

    annotation_text = [" ".join([str(d) for d in l]) + " 1" for l in areas]
    annotation_text = "\n".join(annotation_text)
    annotation_text = f"{image_size}\n{len(areas)}\n{annotation_text}"
    
    with open(output_txt_path, "w") as f:
        f.write(annotation_text)

print("output done.")


# In[ ]:


# import matplotlib.pyplot as plt
# plt.hist([p for p in probs if p >= 0.1], bins=32)


# In[ ]:




