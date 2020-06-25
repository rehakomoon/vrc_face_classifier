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
import math
import random

get_ipython().run_line_magic('matplotlib', 'inline')

data_path = Path.cwd() / "data"
input_dir = data_path / "labeled_data"
output_dir = data_path / "cropped_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


def process(txt_path, png_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    data = [l.split(" ") for l in lines]
    data = [[int(e) for e in d] for d in data]

    load_img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    image_h, image_w, _ = load_img.shape

    if (False):
        plt.imshow(cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB))
        plt.show()

    for i, area in enumerate(data):
        assert len(area) == 5
        x0, y0, x1, y1, _ = area

        crop_image = load_img[y0:y1, x0:x1, :]
        if (False):
            plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if (True):
            filename_prefix = png_path.stem
            file_path = output_dir / f"{filename_prefix}_{i}.png"
            cv2.imwrite(str(file_path), crop_image)

input_path_list = list(input_dir.glob("*.txt"))
input_path_list = [(txt_path, input_dir / (txt_path.stem + ".png")) for txt_path in input_path_list]
input_path_list = [e for e in input_path_list if e[1].exists()]

Parallel(n_jobs=-1, verbose=10)([delayed(process)(*params) for params in input_path_list])
None


# In[ ]:


def read_and_resize(image_path, resize_image_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.resize(image, resize_image_size)
    return image

resize_image_size = (64, 64)
image_path_list = list(output_dir.glob("*.png"))
random.shuffle(image_path_list)
# for debugging
# image_path_list = image_path_list[:100]

num_image_col = int(math.sqrt(len(image_path_list)))
image_path_list = image_path_list[:num_image_col ** 2]

images = [read_and_resize(p, resize_image_size) for p in tqdm(image_path_list)]

pos = list(range(len(image_path_list)))
pos = [(x%num_image_col, math.floor(x/num_image_col)) for x in pos]
pos = [(y*resize_image_size[0], x*resize_image_size[1]) for y, x in pos]

export_image = np.zeros((resize_image_size[1]*num_image_col, resize_image_size[0]*num_image_col, 3), dtype=np.uint8)
for image, p in zip(images, pos):
    export_image[p[0]:p[0]+resize_image_size[0], p[1]:p[1]+resize_image_size[1], :] = image

plt.imshow(cv2.cvtColor(export_image, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite(str(data_path / "faces.png"), export_image)
None


# In[ ]:




