"""
输入：原始视频
输出：带检测框的视频
"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../../")
from object_detection.utils import ops as utils_ops

#if tf.__version__ < '1.5.0':
 # raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

#########################################################
# This is needed to display the images.
#%matplotlib inline

from utils import label_map_util

from utils import visualization_utils as vis_util

import matplotlib;

matplotlib.use('Agg')

# 指定哪张卡
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 随着进程逐渐增加显存占用，而不是一下子占满
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

##########################################################
# What model to download.
# MODEL_NAME = 'training_faster_resnet50/save_model3'
# MODEL_NAME = 'training/save_model4'
MODEL_NAME = './train_output'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('training_faster_resnet50/faster_resnet50', 'new5cls.pbtxt')
PATH_TO_LABELS = os.path.join('../../person_data', 'person_only.pbtxt')

# NUM_CLASSES = 5
NUM_CLASSES = 1

##########################################################
# video to img
# 原始视频
cap = cv2.VideoCapture("./24.mp4")
out_video_dir = './24_output.avi'
fps = 25
# img_size = (1920, 1080)
img_size = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(out_video_dir, fourcc, fps, img_size)

###########################################################
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

###########################################################
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###########################################################
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##############################################################
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images_my'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

###############################################################
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

########################################################
#i = 0
#for image_path in TEST_IMAGE_PATHS:
    #i = i+1
i=0
while True:
    ret, image = cap.read()
    i = i + 1
    print("######################################################")
    print("detecting frame "+str(i))
  # image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
    #image_np = load_image_into_numpy_array(image)
    image_np = image
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=4)
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #cv2.imwrite("/DATACENTER2/jianjun.qiao/tensorflow/models-master/research/object_detection/test_images_my/detect_{}.jpg".format(i), image_np)
    #cv2.imshow('object detection', cv2.resize(image_np, (960,540)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    videoWriter.write(image_np)
    print("####################################################")
    print("frame "+str(i)+" has detected.")
cap.release()
videoWriter.release()
