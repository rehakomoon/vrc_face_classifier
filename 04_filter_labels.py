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
import math

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_png_dir = data_path / "collect_data"

input_dir = data_path / "labeled_data"
output_dir = data_path / "labeled_filtered_data"
positive_list_path = data_path / "filtered_list.txt"
#input_dir = data_path / "merged_data"
#output_dir = data_path / "merged_filtered_data"
#positive_list_path = data_path / "positive_list_merged.txt"

output_dir.mkdir(exist_ok=True)

scale_factor = 0.1


# In[ ]:


with open(positive_list_path, "r") as f:
    target_list = f.readlines()
target_list = [l.strip() for l in target_list]
target_list = [l.split(":") for l in target_list]
target_list = [(f, l.split(",")) for f, l in target_list]
target_list = [(f, [e.split("_") for e in l]) for f, l in target_list]
for e in itertools.chain.from_iterable([[len(e) for e in l] for _, l in target_list]):
    assert(e == 2)

target_list = [(f.strip(), [(int(e[0].strip()), int(e[1].strip())) for e in l]) for f, l in target_list]


# In[ ]:


parameters = [(f"{s}.txt", l) for s, l in target_list]

print(f"#filtered_annotations: {len(target_list)}")


# In[ ]:


def process(params):
    txt_filename, annotations = params

    input_txt_path = input_dir / txt_filename
    output_txt_path = output_dir / txt_filename

    if not input_txt_path.exists():
        print(input_txt_path.name, "does not exist, skipped.")
        return None

    with open(input_txt_path, "r") as f:
        lines = f.readlines()

    image_size, areas = utils.parse_annotation(lines)
    image_w, image_h = image_size

    if len(areas) <= 0:
        return None
    
    if len(areas) <= max([i for i, _ in annotations]):
        print(target_file.name, " annotation is not valid, skipped.")
        return None
    
    areas = [(areas[i], s) for i, s in annotations]
    areas_scaled = []

    for (x0, y0, x1, y1), scale in areas:
        #print(x0, y0, x1, y1)
        orig_w, orig_h = (x1 - x0, y1 - y0)
        assert(orig_h > 0 and orig_w > 0)

        remove_ratio = 1.0 - (1.0 - scale_factor) ** scale
        trim_w = math.ceil(orig_w * remove_ratio * 0.5)
        trim_h = math.ceil(orig_h * remove_ratio * 0.5)
        x0 = min(x0 + trim_w, image_w - 1)
        y0 = min(y0 + trim_h, image_h - 1)
        x1 = max(x1 - trim_w, 0)
        y1 = max(y1 - trim_h, 0)
        #print(x0, y0, x1, y1)
        assert(x0 < x1 and y0 < y1)

        areas_scaled.append((x0, y0, x1, y1))

        #break
    areas = [f"{x0} {y0} {x1} {y1} 1" for x0, y0, x1, y1 in areas_scaled]

    annotations = "\n".join(areas)
    annotations = f"{image_size[0]} {image_size[1]}\n{len(areas)}\n{annotations}"

    with open(output_txt_path, "w") as f:
        f.write(annotations)

[process(p) for p in tqdm(parameters)]
None


# In[ ]:




