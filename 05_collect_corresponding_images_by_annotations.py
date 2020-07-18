#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_txt_dir = data_path / "labeled_filtered_data"
input_png_dir = data_path / "collect_data"
output_dir = data_path / "labeled_filtered_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_path_list = list(input_txt_dir.glob("*.txt"))

# for debugging
# input_path_list = input_path_list[:10]

num_input_txt = len(input_path_list)

parameters = [utils.get_image_path_and_rotation(p) for p in input_path_list]
parameters = zip(input_path_list, parameters)
parameters = [(txt_path, params) for (txt_path, params) in parameters if (params is not None)]
parameters = [(txt_path, input_png_dir / image_filename, rotation) for txt_path, (image_filename, rotation) in parameters]
parameters = [(txt_path, input_png_path, output_dir / f"{input_png_path.stem}_{rotation}.png", rotation) for txt_path, input_png_path, rotation in parameters]

print(f"{num_input_txt-len(parameters)} txt file skipped.")


# In[ ]:


def process(params):
    input_txt_path, input_png_path, output_png_path, rotation = params
    with open(input_txt_path, "r") as f:
        lines = f.readlines()
    image_size, areas = utils.parse_annotation(lines)
    
    image = utils.load_image_with_rotation(input_png_path, rotation)
    image_h, image_w, _ = image.shape
    assert(image_w == image_size[0] and image_h == image_size[1])
    
    cv2.imwrite(str(output_png_path), image)

#[process(params) for params in tqdm(parameters)]
Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in parameters])
print("process done.")


# In[ ]:




