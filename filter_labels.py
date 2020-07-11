#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools

# %matplotlib inline

data_path = Path.cwd() / "data"
input_dir = data_path / "labeled_data"
input_png_dir = data_path / "collect_data"
output_dir = data_path / "labeled_filtered_data"
positive_list_path = data_path / "positive_list.txt"

output_dir.mkdir(exist_ok=True)


# In[ ]:


with open(positive_list_path, "r") as f:
    target_list = f.readlines()
target_list = [l.strip() for l in target_list]
target_list = [l.split(":") for l in target_list]
target_list = [(l[0].strip(), l[1].strip(), l[2].strip()) for l in target_list]
target_list = [(f"{p}_{r}.txt", f"{p}.png", int(r), [int(i.strip()) for i in l.split(",")]) for p, r, l in target_list]


# In[ ]:


def process(params):
    txt_filename, png_filename, rotation, annotations = params
    target_file = input_dir / txt_filename

    if not target_file.exists():
        print(target_file.name, "does not exist, skipped.")
        return None

    with open(target_file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    image_size_str = lines[0]
    num_area = int(lines[1])
    areas = lines[2:]

    if len(areas) != num_area:
        print(target_file.name, "is not valid, skipped.")
        return None

    if num_area <= max(annotations):
        print(target_file.name, " annotation is not valid, skipped.")
        return None

    areas = [areas[i] for i in annotations]

    if len(areas) <= 0:
        return None

    annotations = "\n".join(areas)
    annotations = f"{image_size_str}\n{len(areas)}\n{annotations}"

    return txt_filename, png_filename, rotation, annotations

annotations = [process(p) for p in tqdm(target_list)]
annotations = [a for a in annotations if a]

annotations = [(output_dir / text_filename, input_png_dir / png_filename, r, a) for text_filename, png_filename, r, a in annotations]
annotations = [(output_txt_path, output_dir / (output_txt_path.stem + ".png"), input_png_path, r, a) for output_txt_path, input_png_path, r, a in annotations]

annotations = [p for p in annotations if p[2].exists()]


# In[ ]:


def process_save(params):
    output_txt_path, output_png_path, input_png_path, rotation, annotation = params
    
    image = cv2.imread(str(input_png_path), cv2.IMREAD_COLOR)
    
    rotates = [lambda x: x,
               lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
               lambda x: cv2.rotate(x, cv2.ROTATE_180),
               lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    rotate = rotates[rotation]
    image = rotate(image)

    image_h, image_w, _ = image.shape
    cv2.imwrite(str(output_png_path), image)

    with open(output_txt_path, "w") as f:
        f.write(annotation)

#[process_save(params) for params in annotations]
Parallel(n_jobs=-1, verbose=10)([delayed(process_save)(p) for p in annotations])
print("process done.")


# In[ ]:





# In[ ]:




