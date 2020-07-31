#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed

import utils

#import matplotlib.pyplot as plt
#%matplotlib inline

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

#input_label_dir = data_path / "labeled_data"
#input_png_dir = data_path / "collect_data"
#output_dir = data_path / "face_data"
input_label_dir = data_path / "integrated_data"
input_png_dir = data_path / "collect_data"
output_dir = data_path / "face_data_2"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_path_list = list(input_label_dir.glob("*.txt"))

# for debugging
# input_path_list = input_path_list[:10]

parameters = [utils.get_image_path_and_rotation(p) for p in input_path_list]
parameters = zip(input_path_list, parameters)
parameters = [(txt_path, param) for (txt_path, param) in parameters if param is not None]
parameters = [(txt_path, input_png_dir / png_filename, rotation) for txt_path, (png_filename, rotation) in parameters]

print("#labels:", len(parameters))


# In[ ]:


output_image_size = 128

def process(params):
    input_txt_path, input_png_path, rotation = params
    output_png_filename_prefix = f"{input_png_path.stem}_{rotation}"
    
    with open(input_txt_path, "r") as f:
        lines = f.readlines()
    image_size, areas = utils.parse_annotation(lines)
    
    image = utils.load_image_with_rotation(input_png_path, rotation)
    image_h, image_w, _ = image.shape
    assert(image_w == image_size[0] and image_h == image_size[1])
    
    if (False):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
    area_size_list = []

    for i, area in enumerate(areas):
        x0, y0, x1, y1 = area

        crop_image = image[y0:y1, x0:x1, :]
        crop_h, crop_w, _ = crop_image.shape

        if (False):
            plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if (True):
            crop_image = cv2.resize(crop_image, (output_image_size, output_image_size), interpolation = cv2.INTER_CUBIC)
            file_path = output_dir / f"{output_png_filename_prefix}_{i}.png"
            cv2.imwrite(str(file_path), crop_image)
        
        area_size_list.append((crop_h, crop_w))
    
    return area_size_list

#size_list = [process(params) for params in tqdm(parameters)]
size_list = Parallel(n_jobs=-1, verbose=10)([delayed(process)(params) for params in parameters])
None


# In[ ]:


w_list, h_list = list(zip(*itertools.chain.from_iterable(size_list)))

#plt.hist(w_list, bins=16)

num_scale_up = sum([1 for w in w_list if w < output_image_size])
num_scale_down = len(w_list) - num_scale_up
print(f"total: {len(w_list)}, num_scale_down: {num_scale_down}, num_scale_up: {num_scale_up}, ratio: {num_scale_up / len(w_list)}")


# In[ ]:




