import os
import colorsys
import random
import cv2
import numpy as np
from keras import backend as K

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):    
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_AREA)
    # images/dog.jpg use this is good
    #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    h, w, _ = image.shape

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
                
        # colors: RGB, opencv: BGR
        cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(colors[c])), 6)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(reversed(colors[c])), -1)

        cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
    return image
