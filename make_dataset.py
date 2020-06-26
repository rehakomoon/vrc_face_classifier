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

data_path = Path.cwd() / "data"
input_txt_dir = data_path / "collect_data_label_txt"
input_png_dir = data_path / "collect_data"
output_dir = data_path / "labeled_data"
annotation_output_path = output_dir / "annotation.txt"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_path_list = list(input_txt_dir.glob("*.txt"))

# for debugging
# input_path_list = input_path_list[:10]

input_path_list = [(p, input_png_dir / (p.stem + ".png")) for p in input_path_list]
input_path_list = [(pp, pt) for pt, pp in input_path_list if pp.exists()]

path_list = [(ipp, ipt, output_dir / (f"{i:08}.png"), output_dir / (f"{i:08}.txt")) for i, (ipp, ipt) in enumerate(input_path_list)]

for input_png_path, input_txt_path, output_png_path, output_txt_path in path_list:
    shutil.copy(input_txt_path, output_txt_path)
    shutil.copy(input_png_path, output_png_path)
print("copy done.")


# In[ ]:


annotations = []

for input_png_path, input_txt_path, output_png_path, output_txt_path in path_list:
    annotation = f"#\n{output_png_path.name}\n"
    with open(output_txt_path) as f:
        annotation = annotation + f.read()
    annotations.append(annotation)

annotations = "\n".join(annotations)
with open(annotation_output_path, "w") as f:
    f.write(annotations)
print("output done.")


# In[ ]:




