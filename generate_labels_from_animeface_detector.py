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

# %matplotlib inline

data_path = Path.cwd() / "data"
input_dir = data_path / "collect_data"
output_dir = data_path / "collect_data_label_txt"
cascade_file_path = Path.cwd() / "animeface_detector/lbpcascade_animeface.xml"

output_dir.mkdir(exist_ok=True)


# In[ ]:


def process(image_path):
    load_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    
    cascade = cv2.CascadeClassifier(str(cascade_file_path))
    faces = cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5, minSize = (32, 32))

    if (False):
        plt.imshow(cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB))
        plt.show()

        for (x, y, w, h) in faces:
            cv2.rectangle(load_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        plt.imshow(cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB))
        plt.show()
    
    image_h, image_w, _ = load_img.shape
    #normalized_areas = [(x/image_w, y/image_h, (x+w)/image_w, (y+h)/image_h) for (x, y, w, h) in faces]
    areas = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
    
    if (len(areas) <= 0):
        return
    
    annotation_text = [" ".join([str(d) for d in l]) + " 1" for l in areas]
    annotation_text = "\n".join(annotation_text)
    annotation_text = f"{load_img.shape[1]} {load_img.shape[0]}\n{len(areas)}\n{annotation_text}"

    output_path = output_dir / (image_path.stem + ".txt")
    with open(output_path, "w") as f:
        f.write(annotation_text)

input_image_path_list = list(input_dir.glob("*.png"))

# for debugging
# input_image_path_list = input_image_path_list[:100]

Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in input_image_path_list])
print("process done.")


# In[ ]:




