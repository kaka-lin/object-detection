import os
import time
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_boxes_to_corners
from utils.yolo_utils import *

def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape) # boxes: [y1, x1, y2, x2]

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    ### END CODE HERE ###
    
    return scores, boxes, classes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):    
    # Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    
    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold
    
    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32') # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def image_detection(sess, image_path, image_file):
    # Preprocess your image
    image, image_data = preprocess_image(image_path + image_file, model_image_size = (416, 416))
    
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})

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

def video_detection(sess, image):
    resized_image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})

    colors = generate_colors(class_names)

    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    return image

    
if __name__ == "__main__":
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model_data/tiny-yolo.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
   
    print(output_details)

    # check the type of the input tensor
    #if input_details[0]['dtype'] == np.float32:
    #    floating_model = True

    #scores, boxes, classes = yolo_eval(output_details, image_shape=image_shape)

    # image detection
    #image_file = "dog.jpg"
    #image_path = "images/"
    #image_shape = cv2.imread(image_path + image_file).shape[:2]

    #image, image_data = preprocess_image(image_path + image_file, model_image_size = (416, 416))

    # Run model: start to detect
    # Sets the value of the input tensor.
    #interpreter.set_tensor(input_details[0]['index'], image_data)
    # Invoke the interpreter.
    #interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    #output_data = interpreter.get_tensor(output_details[0]['index'])
