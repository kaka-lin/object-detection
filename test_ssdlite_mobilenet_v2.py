import os
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *

def image_processing(image, model_image_size=300):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (model_image_size, model_image_size))
    image1 = np.expand_dims(image1, axis=0)
    image1 = (2.0 / 255.0) * image1 - 1.0
    image1 = image1.astype('float32')

    return image1

def run_detection(image, interpreter):
    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image)
    # Invoke the interpreter.
    interpreter.invoke()

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))
            
    return out_scores, out_boxes, out_classes

def image_objec_detection(interpreter):
    frame = cv2.imread('images/dog.jpg')
    img = image_processing(frame)
    out_scores, out_boxes, out_classes = run_detection(img, interpreter)

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
    cv2.imwrite(os.path.join("out", "ssdlite_mobilenet_v2_dog.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_object_detection(interpreter):
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        start = time.time()
        ret, frame = camera.read() 

        if ret:
            img = image_processing(frame)
            out_scores, out_boxes, out_classes = run_detection(img, interpreter)
            colors = generate_colors(class_names)
            result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
            end = time.time()

            # fps
            t = end - start
            fps  = "Fps: {:.2f}".format(1 / t)
            cv2.putText(result, fps, (10, 30),
		                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            cv2.imshow('image', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model_data/ssdlite_mobilenet_v2.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # label
    class_names = read_classes('model_data/coco_classes.txt')
            
    #image_objec_detection(interpreter)
    real_time_object_detection(interpreter)
