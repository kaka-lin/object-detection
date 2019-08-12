import os
import time
import tarfile
import glob
import six.moves.urllib as urllib
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *

def run_detection(image_data, sess):
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
    #print('Found {} boxes.'.format(len(out_boxes)))
            
    return out_scores, out_boxes, out_classes

def image_object_detection(image_path, sess, colors):
    image = cv2.imread(image_path)

    image_data = preprocess_image(image, model_image_size=(300,300))
    out_scores, out_boxes, out_classes = run_detection(image_data, sess)

    # Draw bounding boxes on the image file
    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join("out/", "ssd_mobilenet_v1_" + image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_object_detection(sess, colors):
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        start = time.time()
        ret, frame = camera.read() 

        if ret:
            image_data = preprocess_image(frame, model_image_size=(300,300))
            out_scores, out_boxes, out_classes = run_detection(image_data, sess)
            # Draw bounding boxes on the image file
            result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
            end = time.time()

            # fps
            t = end - start
            fps  = "Fps: {:.2f}".format(1 / t)
            # display a piece of text to the frame
            cv2.putText(frame, fps, (10, 30),
		                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Object detection - ssd_mobilenet_v1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    camera.release()
    cv2.destroyAllWindows()

def download_from_url(url, file_name):
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    pbar = tqdm(total=file_size)

    def _progress(block_num, block_size, total_size):
            """callback func
            @block_num: 已經下載的資料塊
            @block_size: 資料塊的大小
            @total_size: 遠端檔案的大小
            """
            pbar.update(block_size)

    filepath, _ = urllib.request.urlretrieve(url, file_name, _progress)
    pbar.close()

def untar_file(file_name, dst):  
    tar_file = tarfile.open(file_name)
    for file in tar_file.getmembers():
        filename = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in filename:
            tar_file.extract(file, dst)

if __name__ == '__main__':
    # What model to download
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    url = download_base + model_file
    
    # Download model to model_data dir
    model_dir = 'model_data'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    file_path = os.path.join(model_dir, model_file)

    # Load a (frozen) Tensorflow model into memory.
    path_to_ckpt = model_dir + '/' + model_name + '/frozen_inference_graph.pb'

    if not os.path.exists(path_to_ckpt):
        download_from_url(url, file_path)
        untar_file(file_path, model_dir)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    
    with detection_graph.as_default():
        with tf.Session() as sess:
            '''
            # image_object_detection
            # Make a list of images
            images = glob.glob('./images/*.jpg')
            for fname in images:
                image_object_detection(fname, sess, colors)
            '''

            # real-time image object detect
            real_time_object_detection(sess, colors)
