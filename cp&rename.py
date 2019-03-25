import os
import shutil

bath = "/Users/zhaoxuyan/Downloads/add_money_output"
output_dir = "/Users/zhaoxuyan/Downloads/output"

def may_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

if __name__ == '__main__':
    may_mkdir(output_dir)
    i = 0
    for root, dirs, files in os.walk(bath):
        print("*"*80)
        print(root)
        # root的第一个是当前目录
        if i == 0:
            i += 1
            continue

        imgs = os.listdir(root)
        for img in imgs:
            if img.endswith(".DS_Store"):
                continue
            new_img_name = str(i) + "_" + img
            print(new_img_name)
            img_path = os.path.join(root, img)
            new_img_path = os.path.join(output_dir,new_img_name)
            shutil.copy(img_path, new_img_path)

        i += 1

