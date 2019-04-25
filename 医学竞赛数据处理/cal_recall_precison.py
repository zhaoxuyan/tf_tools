import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
import cv2
import os

gt_dir = './mask_test'
pre_dir = './dcm_test_result_strict'

gt = []
pre = []

for img in os.listdir(gt_dir):
    gt_img = cv2.imread(os.path.join(gt_dir, img))
    pre_img = cv2.imread(os.path.join(pre_dir, img.split("_mask")[0]+".png"))

    if 255 in gt_img:
        gt.append(1)
    else:
        gt.append(0)

    if 255 in pre_img:
        pre.append(1)
    else:
        pre.append(0)

print("*"*100)
print(metrics.recall_score(gt, pre))
print("*"*100)
print(classification_report(gt, pre))
print("*"*100)