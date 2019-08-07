import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Lambda
from yad2k.models.keras_yolo import yolo_head
from utils.yolo_utils import *
    
if __name__ == "__main__":
    yolo_model = load_model("model_data/tiny-yolo.h5")
    #yolo_model.summary()

    class_names = read_classes("model_data/yolo_coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    # NOTE: Here, we do not include the YOLO head because TFLite does not
    # NOTE: support custom layers yet. Therefore, we'll need to implement
    # NOTE: the YOLO head ourselves.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    output_layer = Lambda(lambda x: K.concatenate(x, axis=-1), name='output')(yolo_outputs) 
    print(output_layer)
    
    new_yolo_model = Model(yolo_model.input, output_layer)
    new_yolo_model.summary()

    #print(yolo_model.output)
    #print(new_yolo_model.output)

    save_model(yolo_model, "model_data/tiny-yolo-tflite.h5", overwrite=True)
