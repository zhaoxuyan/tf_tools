import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image

a = './arterial/mask'
v = './venous/mask'


csv = 'transfer.csv'
iter_csv = pd.read_csv(csv, iterator=True)

# 筛选transer 为 + 的dataframe
df = pd.concat([chunk[chunk['transfer'] == "+"] for chunk in iter_csv])

# Seris to list
# 得到所有阳性病人的id
id_plus = df['id'].tolist()

for file in os.listdir(a):
    person_id = int(file.split("_")[0])
    if person_id in id_plus:
        img = np.array(Image.open(os.path.join(a,file)))
        img = img * 2
        img = Image.fromarray(img.astype(dtype=np.uint8))
        img.save(os.path.join(a,file))

