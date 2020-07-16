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
input_dir = data_path / "collect_data"
output_dir = data_path / "labeled_data"
cascade_file_path = Path.cwd() / "animeface_detector/lbpcascade_animeface.xml"

output_dir.mkdir(exist_ok=True)


# In[ ]:


def process(image_path):
    load_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    images = [load_image,
              cv2.rotate(load_image, cv2.ROTATE_90_CLOCKWISE),
              cv2.rotate(load_image, cv2.ROTATE_180),
              cv2.rotate(load_image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    cascade = cv2.CascadeClassifier(str(cascade_file_path))

    for image_idx, image in enumerate(images):
        filename_prefix = f"{image_path.stem}_{image_idx}"
        output_png_path = output_dir / (filename_prefix + ".png")
        output_txt_path = output_dir / (filename_prefix + ".txt")
        
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)

        faces = cascade.detectMultiScale(gray_img, scaleFactor = 1.01, minNeighbors = 5, minSize = (32, 32))

        if (False):
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

        image_h, image_w, _ = image.shape
        #normalized_areas = [(x/image_w, y/image_h, (x+w)/image_w, (y+h)/image_h) for (x, y, w, h) in faces]
        areas = [(x, y, x+w, y+h) for (x, y, w, h) in faces]

        if (len(areas) <= 0):
            continue

        annotation_text = [" ".join([str(d) for d in l]) + " 1" for l in areas]
        annotation_text = "\n".join(annotation_text)
        annotation_text = f"{image.shape[1]} {image.shape[0]}\n{len(areas)}\n{annotation_text}"
        
        #cv2.imwrite(str(output_png_path), image)

        with open(output_txt_path, "w") as f:
            f.write(annotation_text)

input_image_path_list = list(input_dir.glob("*.png"))

# for debugging
# input_image_path_list = input_image_path_list[:100]

Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in input_image_path_list])
print("process done.")


# In[ ]:




