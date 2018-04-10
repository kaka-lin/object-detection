import os
import time
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yolov3.model import yolo_eval
from utils.yolo_utils import *

def image_detection(sess, image_path, image_file):
    # Preprocess your image
    image, image_data = preprocess_image(image_path + image_file, model_image_size = (416, 416))
    
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolov3.input:image_data, K.learning_phase():0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    #image.save(os.path.join("out", image_file), quality=90)
    cv2.imwrite(os.path.join("out", image_file), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    return out_scores, out_boxes, out_classes

if __name__ == "__main__":
    sess = K.get_session()

    yolov3 = load_model("model_data/yolov3.h5")
    #yolov3.summary()
    
    class_names = read_classes("model_data/yolo_coco_classes.txt")
    anchors = read_anchors("model_data/yolov3_anchors.txt")

    # image detection
    image_file = "dog.jpg"
    image_path = "images/"
    image_shape = np.float32(cv2.imread(image_path + image_file).shape[:2])

    scores, boxes, classes = yolo_eval(yolov3.output, anchors, len(class_names), image_shape=image_shape)
    out_scores, out_boxes, out_classes = image_detection(sess, image_path, image_file)
    