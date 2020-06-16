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
input_dir = data_path / "collect_data"
output_dir = data_path / "labeled_data"
cascade_file_path = Path.cwd() / "animeface_detector/lbpcascade_animeface.xml"

output_dir.mkdir(exist_ok=True)


# In[ ]:


def extract_face_areas(image_path):
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
    normalized_areas = [(x/image_w, y/image_h, (x+w)/image_w, (y+h)/image_h) for (x, y, w, h) in faces]
    
    return normalized_areas

def output_image_with_annotation(params):
    output_idx, (image_path, annotation) = params
    annotation_text = [" ".join([str(d) for d in l]) for l in annotation]
    annotation_text = ["0 " + s for s in annotation_text]
    annotation_text = "\n".join(annotation_text)
    # print(annotation_text)

    filename_prefix = f"{output_idx:08}"
    txt_file_path = output_dir / (filename_prefix + ".txt")
    png_file_path = output_dir / (filename_prefix + ".png")

    with open(txt_file_path, "w") as f:
        f.write(annotation_text)
    shutil.copy(image_path, png_file_path)

input_image_path_list = list(input_dir.glob("*.png"))

# for debugging
input_image_path_list = input_image_path_list[:10000]

extracted = Parallel(n_jobs=-1, verbose=10)([delayed(extract_face_areas)(p) for p in input_image_path_list])
extracted = zip(input_image_path_list, extracted)
print("face area extraction done.")
extracted = [e for e in extracted if len(e[1]) > 0]
extracted = list(enumerate(extracted))

for param in tqdm(extracted):
    output_image_with_annotation(param)
print("save done.")

