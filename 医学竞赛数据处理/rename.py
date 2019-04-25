import os
import shutil

root = './data_rename'
output = './output'

for i in range(1001, 1109):
    dir_ = os.path.join(root, str(i)) + "/arterial phase"
    dir__ = os.path.join(root, str(i)) + "/venous phase"

    if os.path.isdir(dir_):
        for file in os.listdir(dir_):
            old = os.path.join(dir_, file)
            new = os.path.join(output, str(i)+"_"+file)
            shutil.copyfile(old, new)

    if os.path.isdir(dir__):
        for file in os.listdir(dir__):
            old = os.path.join(dir__, file)
            new = os.path.join(output, str(i)+"_"+file)
            shutil.copyfile(old, new)