#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_png_dir = data_path / "collect_data"

input_dir = data_path / "labeled_data"
output_dir = data_path / "labeled_filtered_data"
positive_list_path = data_path / "positive_list.txt"
#input_dir = data_path / "merged_data"
#output_dir = data_path / "merged_filtered_data"
#positive_list_path = data_path / "positive_list_merged.txt"

output_dir.mkdir(exist_ok=True)


# In[ ]:


with open(positive_list_path, "r") as f:
    target_list = f.readlines()
target_list = [l.strip() for l in target_list]
target_list = [l.split(":") for l in target_list]
target_list = [(l[0].strip(), l[1].strip(), l[2].strip()) for l in target_list]
target_list = [(f"{p}_{r}.txt", f"{p}.png", int(r), [int(i.strip()) for i in l.split(",")]) for p, r, l in target_list]

print(f"#filtered_annotations: {len(target_list)}")


# In[ ]:


def process(params):
    txt_filename, png_filename, rotation, annotations = params
    input_txt_path = input_dir / txt_filename
    output_txt_path = output_dir / txt_filename

    if not input_txt_path.exists():
        print(input_txt_path.name, "does not exist, skipped.")
        return None

    with open(input_txt_path, "r") as f:
        lines = f.readlines()
        
    image_size, areas = utils.parse_annotation(lines)
    
    if len(areas) <= max(annotations):
        print(target_file.name, " annotation is not valid, skipped.")
        return None
    
    areas = [areas[i] for i in annotations]
    areas = [f"{x0} {y0} {x1} {y1} 1" for x0, y0, x1, y1 in areas]
    
    if len(areas) <= 0:
        return None

    annotations = "\n".join(areas)
    annotations = f"{image_size[0]} {image_size[1]}\n{len(areas)}\n{annotations}"
    
    with open(output_txt_path, "w") as f:
        f.write(annotations)

[process(p) for p in tqdm(target_list)]
None


# In[ ]:




