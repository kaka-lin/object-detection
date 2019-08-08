import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import plot_model
from utils.yolo_utils import *

def yolo_head(feats, anchors, num_classes):
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))
    
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return [box_confidence, box_xy, box_wh, box_class_probs]

if __name__ == "__main__":
    yolo_model = load_model("model_data/tiny_yolo.h5")
    #yolo_model.summary()

    class_names = read_classes("model_data/yolo_coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    # NOTE: Here, we do not include the YOLO head because TFLite does not
    # NOTE: support custom layers yet. Therefore, we'll need to implement
    # NOTE: the YOLO head ourselves.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    output_layer = Lambda(lambda x: K.concatenate(x, axis=-1), name='output')(yolo_outputs)
    #expand_layer = Lambda(lambda x: K.expand_dims(x, axis=0), name='expand_output')(yolo_model.output) 

    new_yolo_model = Model(yolo_model.input, output_layer)
    
    new_yolo_model.summary()
    plot_model(new_yolo_model, to_file="model.png", show_shapes=True)

    save_model(yolo_model, "model_data/tiny_yolo_tflite.h5", overwrite=True)


