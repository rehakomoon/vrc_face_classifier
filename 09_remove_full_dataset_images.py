#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import itertools

data_path = Path.cwd() / "data"
#data_path = Path.cwd() / "data_tiny"

process_dir = data_path / "inference_data"


# In[ ]:


image_path_list = list(process_dir.glob("*.png"))

#[p.unlink() for p in tqdm(image_path_list)]
Parallel(n_jobs=-1, verbose=10)([delayed(lambda x: x.unlink())(p) for p in image_path_list])
print("process done.")


# In[ ]:




