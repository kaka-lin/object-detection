import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("model_data/tiny_yolo_tflite.h5")
#converter.post_training_quantize = True
tflite_model = converter.convert()
open("model_data/tiny_yolo.tflite", "wb").write(tflite_model)
