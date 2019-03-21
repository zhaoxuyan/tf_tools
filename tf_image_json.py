"""
输入：图片 dir
输出：json dir
"""
import time
start = time.time()
import numpy as np
import os
import tensorflow as tf

from PIL import Image
from object_detection.utils import label_map_util
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Tensorflow默认的logging information太多，看着糟心
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 显卡配置
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# 随着进程逐渐增加显存占用，而不是一下子占满
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

###############################################
PATH_TO_CKPT = './frozen_inference_graph_map40.pb'
PATH_TO_LABELS = '../../data/vehicle_label_map.pbtxt'

NUM_CLASSES = 12            # classes number

# PATH_TO_TEST_IMAGES_DIR = os.getcwd()+'/test_images_my'               # image path

confident = 0.5  #only scores>confident, the target will output
###############################################

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# os.chdir(PATH_TO_TEST_IMAGES_DIR)
for aaa in range(1,21):
    if aaa in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20]:
       continue
    # PATH_TO_TEST_IMAGES_DIR = '/home/xuyan.zhao/frames/c'+str(aaa)
    PATH_TO_TEST_IMAGES_DIR = '/home/xuyan.zhao/c15'
    imgs = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    imgs.sort(key= lambda x:int(x[:-4]))
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1, 8000) ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        n = 1
        # for image_path in TEST_IMAGE_PATHS:     # start dtecting
        for img in imgs:
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, img)
            image = Image.open(image_path)          # read image
            width, height = image.size
            image_np = np.array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            s_boxes = boxes[scores > confident]
            s_classes = classes[scores > confident]
            s_scores = scores[scores > confident]
            list_classes = []
            for i in range(len(s_classes)):
                name = image_path.split("/")[-1]
                # name = image_path.split("\\")[-1].split('.')[0]   # without .jpg
                # ymin = s_boxes[i][0] * height  # ymin
                # xmin = s_boxes[i][1] * width  # xmin
                # ymax = s_boxes[i][2] * height  # ymax
                # xmax = s_boxes[i][3] * width  # xmax
                ymin = s_boxes[i][0]  # ymin
                xmin = s_boxes[i][1]  # xmin
                ymax = s_boxes[i][2]  # ymax
                xmax = s_boxes[i][3]  # xmax
                score = s_scores[i]
                if s_classes[i] in category_index.keys():
                    class_name = category_index[s_classes[i]]['name']  # get class name
                    print("====================>", s_classes[i])
                list_classes.append(class_name)
                print("name:", name)
                print("ymin:", ymin)
                print("xmin:", xmin)
                print("ymax:", ymax)
                print("xmax:", xmax)
                print("score:", score)
                print("class:", class_name)
                print("################")
            write_dict = {'boxes':s_boxes, 'classes':list_classes}
            json_str = json.dumps(write_dict, cls=NumpyEncoder)
            new_dict = json.loads(json_str)
            with open("./json"+str(aaa)+"_map40/"+str(n)+".json","w") as f:
                json.dump(new_dict,f)
                print("load file...")
            n = n + 1

    end = time.time()
    print("Execution Time: ", end-start)
