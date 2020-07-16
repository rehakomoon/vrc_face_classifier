#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed

#%matplotlib inline

data_path = Path.cwd() / "data"
input_label_dir = data_path / "labeled_data"
input_png_dir = data_path / "collect_data"
output_dir = data_path / "face_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_path_list = list(input_label_dir.glob("*.txt"))

# for debugging
# input_path_list = input_path_list[:10]

print("#labels:", len(input_path_list))


# In[ ]:


parameters = input_path_list
parameters = [(p, p.stem.split("_")) for p in parameters]
parameters = [p for p in parameters if len(p[1]) == 3]
parameters = [(txt_path, input_png_dir / f"{p[0]}_{p[1]}.png", f"{txt_path.stem}", int(p[2])) for txt_path, p in parameters]

# parameters
# input_txt_path, input_png_path, output_png_filename_prefix, rotation


# In[ ]:


image_size = 256

rotates = [lambda x: x,
           lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
           lambda x: cv2.rotate(x, cv2.ROTATE_180),
           lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)]

def process(params):
    input_txt_path, input_png_path, output_png_filename_prefix, rotation = params
    with open(input_txt_path, "r") as f:
        lines = f.readlines()
    lines = lines[2:]
    lines = [l.strip() for l in lines]
    data = [l.split(" ") for l in lines]
    data = [[int(e) for e in d] for d in data]

    image = cv2.imread(str(input_png_path), cv2.IMREAD_COLOR)
    image = rotates[rotation](image)
    
    if (False):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
    image_size_list = []

    for i, area in enumerate(data):
        assert len(area) == 5
        x0, y0, x1, y1, _ = area

        crop_image = image[y0:y1, x0:x1, :]
        crop_h, crop_w, _ = crop_image.shape

        if (False):
            plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if (True):
            crop_image = cv2.resize(crop_image, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
            file_path = output_dir / f"{output_png_filename_prefix}_{i}.png"
            cv2.imwrite(str(file_path), crop_image)
        
        image_size_list.append((crop_h, crop_w))
    
    return image_size_list

#size_list = [process(params) for params in tqdm(parameters)]
size_list = Parallel(n_jobs=-1, verbose=10)([delayed(process)(params) for params in parameters])


# In[ ]:


size_list = itertools.chain.from_iterable(size_list)
w_list, h_list = list(zip(*size_list))

# import matplotlib.pyplot as plt
# plt.hist(w_list, bins=16)

num_scale_up = sum([1 for w in w_list if w < image_size])
num_scale_down = len(w_list) - num_scale_up
print(f"total: {len(w_list)}, num_scale_down: {num_scale_down}, num_scale_up: {num_scale_up}, ratio: {num_scale_up / len(w_list)}")


# In[ ]:




