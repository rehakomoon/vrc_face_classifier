#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_txt_dir_1 = data_path / "labeled_data"
input_txt_dir_2 = data_path / "inference_data"
output_dir = data_path / "merged_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_txt_list_1 = input_txt_dir_1.glob("*.txt")
input_txt_list_2 = input_txt_dir_2.glob("*.txt")
input_txt_list_1 = [p.stem for p in input_txt_list_1]
input_txt_list_2 = [p.stem for p in input_txt_list_2]

# for debugging
# input_txt_list_1 = input_txt_list_1[:10000]
# input_txt_list_2 = input_txt_list_2[:10000]


# In[ ]:


input_txt_list_shared = [p for p in tqdm(input_txt_list_1) if (p in input_txt_list_2)]
input_txt_list_1_unique = [p for p in tqdm(input_txt_list_1) if (p not in input_txt_list_shared)]
input_txt_list_2_unique = [p for p in tqdm(input_txt_list_2) if (p not in input_txt_list_shared)]

print("#1_unique: ", len(input_txt_list_1_unique),
      ", #2_unique: ", len(input_txt_list_2_unique),
      ", #shared: ", len(input_txt_list_shared))

copy_file_list_1 = [(input_txt_dir_1 / (p + ".txt"), output_dir / (p + ".txt")) for p in input_txt_list_1_unique]
copy_file_list_2 = [(input_txt_dir_2 / (p + ".txt"), output_dir / (p + ".txt")) for p in input_txt_list_2_unique]
copy_file_list = copy_file_list_1 + copy_file_list_2
merge_file_list = [(input_txt_dir_1 / (p + ".txt"), input_txt_dir_2 / (p + ".txt"), output_dir / (p + ".txt")) for p in input_txt_list_shared]


# In[ ]:


#[shutil.copyfile(*params) for params in tqdm(copy_file_list)]
Parallel(n_jobs=-1, verbose=10)([delayed(lambda x: shutil.copyfile(*x))(p) for p in copy_file_list])
print("copy done.")


# In[ ]:


def process(params):
    input_txt_path_1, input_txt_path_2, output_txt_path = params
    
    with open(input_txt_path_1, "r") as f:
        lines = f.readlines()
    image_size_1, areas_1 = utils.parse_annotation(lines)
    
    with open(input_txt_path_2, "r") as f:
        lines = f.readlines()
    image_size_2, areas_2 = utils.parse_annotation(lines)
    
    assert(image_size_1 == image_size_2)
    image_size = image_size_1
    
    areas = areas_1 + areas_2
    num_areas = len(areas)
    
    areas = [f"{x0} {y0} {x1} {y1} 1" for x0, y0, x1, y1 in areas]
    areas = "\n".join(areas)
    
    annotation = f"{image_size[0]} {image_size[1]}\n{num_areas}\n{areas}"
    
    with open(output_txt_path, "w") as f:
        f.write(annotation)

#[process(params) for params in tqdm(merge_file_list)]
Parallel(n_jobs=-1, verbose=10)([delayed(process)(p) for p in merge_file_list])
print("merge done.")


# In[ ]:


print("output done.")


# In[ ]:




