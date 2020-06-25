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

#get_ipython().run_line_magic('matplotlib', 'inline')

data_path = Path.cwd() / "data"
input_dir = data_path / "raw_data"
output_dir = data_path / "collect_data"

output_dir.mkdir(exist_ok=True)

target_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
target_paths = [d.glob("*.png") for d in target_dirs]
target_paths = list(itertools.chain.from_iterable(target_paths))


# In[ ]:


"""
for load_path in tqdm(target_paths):
    load_image = cv2.imread(str(load_path), cv2.IMREAD_COLOR)
    assert load_image.shape[2] == 3
    
    images = [load_image, cv2.flip(load_image, 1)]
    images = [[image,
               cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
               cv2.rotate(image, cv2.ROTATE_180),
               cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)] for image in images]
    images = list(itertools.chain.from_iterable(images))
    
    if (False):
        for image in images:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

    output_path_prefix = load_path.parent.stem + "_" + load_path.stem.replace(".", "_")
    for i, image in enumerate(images):
        output_path = output_dir / f"{output_path_prefix}_{i}.png"
        cv2.imwrite(str(output_path), image)
    # break
"""


# In[ ]:


def process(load_path):
    load_image = cv2.imread(str(load_path), cv2.IMREAD_COLOR)
    if (load_image is None):
        return
    assert load_image.shape[2] == 3

    load_image = cv2.resize(load_image, (load_image.shape[1] // 2, load_image.shape[0] // 2))
    
    images = [load_image, cv2.flip(load_image, 1)]
    images = [[image,
               cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
               cv2.rotate(image, cv2.ROTATE_180),
               cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)] for image in images]
    images = list(itertools.chain.from_iterable(images))
    
    if (False):
        for image in images:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

    output_path_prefix = load_path.parent.stem + "_" + load_path.stem.replace(".", "_")
    for i, image in enumerate(images):
        output_path = output_dir / f"{output_path_prefix}_{i}.png"
        cv2.imwrite(str(output_path), image)

# for debugging
#target_paths = target_paths[:1000]

Parallel(n_jobs=-1, verbose=10)([delayed(process)(target_path) for target_path in target_paths])
None


# In[ ]:




