#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools

import utils

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

input_txt_dirs = [data_path / "labeled_data", data_path / "inference_data"]
output_dir = data_path / "merged_data"

output_dir.mkdir(exist_ok=True)


# In[ ]:


input_txt_lists = [(d.glob("*.txt"), i) for i, d in enumerate(input_txt_dirs)]
input_txt_lists = [[(p.stem, i) for p in l] for l, i in input_txt_lists]
input_txt_lists = itertools.chain.from_iterable(input_txt_lists)
input_txt_lists = [(p, i) for (p, i) in input_txt_lists if len(p.split("_")) == 3]

#input_txt_lists = list(input_txt_lists)
#input_txt_list_2 = [p.stem for p in input_txt_list_2]
#input_txt_lists


# In[ ]:


keys = {k for k, _ in input_txt_lists}
annotation_txt_lists = {k: [] for k in keys}

for filename, idx in input_txt_lists:
    annotation_txt_lists[filename].append(idx)

annotation_txt_lists = [(k, sorted(list(set(v)))) for k, v in annotation_txt_lists.items()]

input_txt_list_unique = [(filename+".txt", input_txt_dirs[idx_list[0]]) for filename, idx_list in tqdm(annotation_txt_lists) if len(idx_list) == 1]
input_txt_list_shared = [(filename+".txt", [input_txt_dirs[i] for i in idx_list]) for filename, idx_list in tqdm(annotation_txt_lists) if len(idx_list) > 1]

copy_file_list = [(src_dir / filename, output_dir / filename) for filename, src_dir in input_txt_list_unique]
merge_file_list = [([d / filename for d in src_dirs], output_dir / filename) for filename, src_dirs in input_txt_list_shared]


# In[ ]:


#[shutil.copyfile(*params) for params in tqdm(copy_file_list)]
Parallel(n_jobs=-1, verbose=10)([delayed(lambda x: shutil.copyfile(*x))(p) for p in copy_file_list])
print("copy done.")


# In[ ]:


def process(params):
    input_txt_paths, output_txt_path = params
    assert(len(input_txt_paths) > 0)
    
    image_size_list = []
    areas_list = []
    
    for input_txt_path in input_txt_paths:
        with open(input_txt_path, "r") as f:
            lines = f.readlines()
        image_size, areas = utils.parse_annotation(lines)
        image_size_list.append(image_size)
        areas_list.append(areas)
    
    image_size = image_size_list[0]
    for s in image_size_list:
        assert(s == image_size)
    
    areas = list(itertools.chain.from_iterable(areas_list))
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




