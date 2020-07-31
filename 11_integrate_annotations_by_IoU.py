#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_dir = data_path / "merged_data"
output_dir = data_path / "integrated_data"

output_dir.mkdir(exist_ok=True)


# In[2]:


parameters = input_dir.glob("*.txt")
parameters = [p for p in parameters if len(p.stem.split("_")) == 3]
parameters = [(p, output_dir / p.name) for p in parameters]


# In[3]:


iou_merge_threshold = 0.8


# In[4]:


def process(params):
    input_path, output_path = params
    #print(input_path)
    
    with open(input_path, "r") as f:
        lines = f.readlines()
    image_size, areas = utils.parse_annotation(lines)
    
    input_areas = [(x0, y0, x1, y1, (x1-x0)*(y1-y0)) for x0, y0, x1, y1 in areas]
    output_areas = []

    for x0i, y0i, x1i, y1i, ai in input_areas:
        need_to_append = True
        for idx, (x0o, y0o, x1o, y1o, ao, no) in enumerate(output_areas):
            x0u, y0u, x1u, y1u = (max(x0i, x0o), max(y0i, y0o), min(x1i, x1o), min(y1i, y1o))
            if (x1u - x0u <= 0 or y1u - y0u <= 0):
                continue
            au = (x1u - x0u) * (y1u - y0u)
            iou = 2 * au / (ai + ao)
            assert(iou <= 1)
            if (iou >= iou_merge_threshold):
                no_new = no + 1
                x0 = max(0, int((x0o * no + x0i) / no_new))
                y0 = max(0, int((y0o * no + y0i) / no_new))
                x1 = min(image_size[0] - 1, int((x1o * no + x1i) / no_new))
                y1 = min(image_size[1] - 1, int((y1o * no + y1i) / no_new))
                assert(x1 - x0 > 0)
                assert(y1 - y0 > 0)
                a = (x1 - x0) * (y1 - y0)
                output_areas[idx] = (x0, y0, x1, y1, a, no_new)
                need_to_append = False
                break
        if need_to_append:
            output_areas.append((x0i, y0i, x1i, y1i, ai, 1))

    areas = [f"{x0} {y0} {x1} {y1} 1" for x0, y0, x1, y1, _, _ in output_areas]
    num_areas = len(areas)
    areas = "\n".join(areas)
    
    annotation = f"{image_size[0]} {image_size[1]}\n{num_areas}\n{areas}"
    #print(len(input_areas), num_areas)
    
    with open(output_path, "w") as f:
        f.write(annotation)

#[process(params) for params in tqdm(parameters)]
Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in parameters])
print("output done.")


# In[ ]:




