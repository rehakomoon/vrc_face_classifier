#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_dir = data_path / "collect_data"
output_dir = data_path / "inference_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


def process(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    images = [image,
              cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
              cv2.rotate(image, cv2.ROTATE_180),
              cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    annotations = []

    for image_idx, image in enumerate(images):
        png_filename = f"{image_path.stem}_{image_idx}.png"
        output_path = output_dir / png_filename

        image_h, image_w, _ = image.shape
        annotation_text = f"{png_filename}\n{image_w} {image_h}\n0"
        annotations.append(annotation_text)
        
        cv2.imwrite(str(output_path), image)
    
    return annotations

input_image_path_list = list(input_dir.glob("*.png"))

# for debugging
# input_image_path_list = input_image_path_list[:100]

#annotations = [process(p) for p in tqdm(input_image_path_list)]
annotations = Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in input_image_path_list])
annotations = list(itertools.chain.from_iterable(annotations))

annotation_path = output_dir / "annotation.txt"
annotations = "#\n" + "\n#\n".join(annotations)
with open(annotation_path, "w") as f:
    f.write(annotations)

print("process done.")


# In[ ]:




