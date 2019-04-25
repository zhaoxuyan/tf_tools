import os
import shutil

output1 = './arterial'
output2 = './venous'

dcm = './dcm'
mask = './mask'

a_dcm = os.path.join(output1, "dcm")
a_mask = os.path.join(output1, "mask")

v_dcm = os.path.join(output2, "dcm")
v_mask = os.path.join(output2, "mask")


for file in os.listdir(dcm):
    if file.split("_")[1].startswith("1"):
        old = os.path.join(dcm, file)
        new = os.path.join(a_dcm, file)
        shutil.copyfile(old, new)
    elif file.split("_")[1].startswith("2"):
        old = os.path.join(dcm, file)
        new = os.path.join(v_dcm, file)
        shutil.copyfile(old, new)

for file in os.listdir(mask):
    if file.split("_")[1].startswith("1"):
        old = os.path.join(mask, file)
        new = os.path.join(a_mask, file)
        shutil.copyfile(old, new)
    elif file.split("_")[1].startswith("2"):
        old = os.path.join(mask, file)
        new = os.path.join(v_mask, file)
        shutil.copyfile(old, new)