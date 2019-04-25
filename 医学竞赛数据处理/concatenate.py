import cv2
import os
import numpy as np

dcm_dir = './dcm_test'
mask_gt_dir = './mask_test'
mask_predict_dir = './dcm_test_result_strict'

concat_dir = './result_strict'

for img in os.listdir(dcm_dir):
    print(img)
    dcm_name = img
    mask_gt_name = dcm_name[:-4] + "_mask.png"
    mask_predict_name = dcm_name[:-4] + ".png"

    dcm = cv2.imread(os.path.join(dcm_dir, dcm_name))
    mask_gt = cv2.imread(os.path.join(mask_gt_dir, mask_gt_name))
    mask_predict = cv2.imread(os.path.join(mask_predict_dir, mask_predict_name))

    # print(dcm.shape)
    # print(mask_gt.shape)
    # print(mask_predict.shape)


    # break

    result = np.concatenate((dcm, mask_gt, mask_predict), axis=1)
    cv2.imwrite(os.path.join(concat_dir, mask_predict_name), result)

