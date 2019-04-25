import os
import shutil

input_dir = './Patient-NAME'
output = './dcm'

for (root, dirs, files) in os.walk(input_dir):
    # print(root)
    for f in files:
        print(f)
        if f.endswith(".jpg"):
            old = os.path.join(root, f)
            new = os.path.join(output,f)
            shutil.copyfile(old,new)