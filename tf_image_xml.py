"""
输入：图片dir,frozen_inference_graph.pb
输出：图片对应的xml dir
"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from xml.dom.minidom import Document
from collections import defaultdict
from io import StringIO
from PIL import Image
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

sys.path.append("../../")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.12.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
import time

def tic():
    globals()['tt'] = time.time()

def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.time()-globals()['tt']))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


PATH_TO_CKPT = './train_output/frozen_inference_graph.pb'
PATH_TO_LABELS = '../../person_data/person_only.pbtxt'

NUM_CLASSES = 1

class Detector():
    def __init__(self, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

                self.session = tf.Session(config=config)

                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                self.tensor_dict = {}
                for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in self.tensor_dict:
                    detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])

                    real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, frame.shape[0], frame.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    def process_img(self, img):
        tic()
        output_dict = self.session.run(self.tensor_dict,
                       feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        toc()
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        mask = output_dict['detection_scores']>0.5
        return output_dict['detection_classes'][mask],output_dict['detection_scores'][mask],output_dict['detection_boxes'][mask]


def may_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

if __name__ == '__main__':
    # class_name = ['person', 'motorbike', 'car', 'SUV', 'bus', 'microbus', 'pickup', 'truck', 'tanker', 'tractor', 'engineeringvan', 'tricycle']
    class_name = ['person']

    d = Detector(PATH_TO_CKPT,PATH_TO_LABELS, NUM_CLASSES)
    # j=0
    img_dir = "./money_person"
    xml_output_dir = "./money_person_xml"
    may_mkdir(xml_output_dir)

    for img_filename in os.listdir(img_dir):

        img = cv2.imread(os.path.join(img_dir, img_filename))
        width_origin = img.shape[1]
        height_origin = img.shape[0]
        classes,scores,boxes=d.process_img(img)
        # print(j)
        # j=j+1
        #生成xml
        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)

        #folder
        folder = doc.createElement('folder')
        annotation.appendChild(folder)
        folder.appendChild(doc.createTextNode("money_person"))

        #filename
        filename = doc.createElement('filename')
        annotation.appendChild(filename)
        filename.appendChild(doc.createTextNode(img_filename))

        #path
        path = doc.createElement('path')
        annotation.appendChild(path)
        path.appendChild(doc.createTextNode("/disk3/xuyan.zhao/models/research/object_detection/exp/faster_rcnn_resnet101_people"))

        #source
        source = doc.createElement('source')
        annotation.appendChild(source)
        database = doc.createElement('database')
        source.appendChild(database)
        database.appendChild(doc.createTextNode("Unknown"))

        #size
        size = doc.createElement('size')
        annotation.appendChild(size)
        width = doc.createElement('width')
        height = doc.createElement('height')
        depth = doc.createElement('depth')
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)

        # 图片尺寸要改
        width.appendChild(doc.createTextNode(str(width_origin)))
        height.appendChild(doc.createTextNode(str(height_origin)))
        depth.appendChild(doc.createTextNode("3"))

        #segmented
        segmented = doc.createElement('segmented')
        annotation.appendChild(segmented)
        segmented.appendChild(doc.createTextNode("0"))

        #===============================================================
        #object
        i=0
        while i<len(classes):
            object = doc.createElement('object')
            annotation.appendChild(object)
            name = doc.createElement('name')
            pose = doc.createElement('pose')
            truncated = doc.createElement('truncated')
            difficult = doc.createElement('difficult')
            bndbox = doc.createElement('bndbox')
            score = doc.createElement('score')
            object.appendChild(name)
            object.appendChild(pose)
            object.appendChild(truncated)
            object.appendChild(difficult)
            object.appendChild(bndbox)
            object.appendChild(score)
            name.appendChild(doc.createTextNode(class_name[classes[i]-1]))
            pose.appendChild(doc.createTextNode("Unspecified"))
            truncated.appendChild(doc.createTextNode("0"))
            difficult.appendChild(doc.createTextNode("0"))
            score.appendChild(doc.createTextNode(str(scores[i])))
            #bndbox
            xmin = doc.createElement('xmin')
            ymin = doc.createElement('ymin')
            xmax = doc.createElement('xmax')
            ymax = doc.createElement('ymax')
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)
            xmin.appendChild(doc.createTextNode(str(int(boxes[i][1]*width_origin))))
            ymin.appendChild(doc.createTextNode(str(int(boxes[i][0]*height_origin))))
            xmax.appendChild(doc.createTextNode(str(int(boxes[i][3]*width_origin))))
            ymax.appendChild(doc.createTextNode(str(int(boxes[i][2]*height_origin))))
            i=i+1
        with open(os.path.join(xml_output_dir, img_filename[:-3]+"xml"), 'wb') as f:
            f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
            print("done:",img_filename)