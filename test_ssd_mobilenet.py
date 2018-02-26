import os
import time
import tarfile
import six.moves.urllib as urllib
import threading
import cv2
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *

def object_detection(image, image_data, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
           
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')

    boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                    feed_dict={image_tensor: image_data})
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), image_name))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
            
    return image
            
def single_image_detect(image_name, image_file, detection_graph):
    image = cv2.imread(image_file) # (636, 1024, 3)

    image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #resized_image = cv2.resize(image, tuple(reversed((300, 300))), interpolation=cv2.INTER_AREA)
    #resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    #image_data = resized_image

    image_data_expanded = np.expand_dims(image_data, axis=0)
    image = object_detection(image, image_data_expanded, detection_graph)

    # Save the predicted bounding box on the image
    cv2.imwrite(os.path.join("out", image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_image_detect(detection_graph):
    with detection_graph.as_default():
        with tf.Session() as sess:
            camera = cv2.VideoCapture(0)

            while camera.isOpened():
                start = time.time()
                ret, frame = camera.read() 

                if ret:
                    image = frame
                    image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_data_expanded = np.expand_dims(image_data, axis=0)
                    image = object_detection(image, image_data_expanded, sess)
                    end = time.time()

                    # fps
                    t = end - start
                    fps  = "Fps: {:.2f}".format(1 / t)
                    # display a piece of text to the frame
                    cv2.putText(image, fps, (10, 30),
		                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)	

                    cv2.imshow('image', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            camera.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # What model to download
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    
    # Download model to model_data dir
    model_dir = 'model_data'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, model_file)

    opener = urllib.request.URLopener()
    opener.retrieve(download_base + model_file, model_path)

    # Untar model
    tar_file = tarfile.open(model_path)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, model_dir)

    # Load a (frozen) Tensorflow model into memory.
    path_to_ckpt = model_dir + '/' + model_name + '/frozen_inference_graph.pb'
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # label
    class_names = read_classes('model_data/coco_classes.txt')
    
    '''
    # image object detect
    image_dir = 'images'
    image_names = ['image{}.jpg'.format(i) for i in range(1, 4)]
    for image_name in image_names:
        image_file = os.path.join(image_dir, image_name)
        single_image_detect(image_name, image_file, detection_graph)
    '''

    # real-time image object detect
    real_time_image_detect(detection_graph)
  