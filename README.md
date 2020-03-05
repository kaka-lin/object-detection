# Object Detection

* training:

    - [yolov2-tf2](https://github.com/kaka-lin/yolov2-tf2)
    - [yolov3-tf2](https://github.com/kaka-lin/yolov3-tf2)

* model (Inference):

    - [tiny-YOLOv2](#tiny-yolo)
    - [YOLOv3](#yolov3)
    - [SSD-MobileNet v1](#ssd-mobilenet-v1)
    - [SSDLite-MobileNet v2 (tflite)](#ssdlite-mobilenet-v2)

## Usage

<span id="tiny-yolo"></span>
### 1. tiny-YOLOv2

* download the [tiny-yolo](https://drive.google.com/file/d/14-5ZojD1HSgMKnv6_E3WUcBPxaVm52X2/view?usp=sharing) file and put it to model_data file

```baash 
$ python3 test_tiny_yolo.py 
```

<span id="yolov3"></span>
### 2. YOLOv3

* download the [yolov3](https://drive.google.com/open?id=1vdD9TPiTWqvPxtCXdbVSKKksSdu0j_Hn) file and put it to model_data file

```baash 
$ python3 test_yolov3.py 
```

<span id="ssd-mobilenet-v1"></span>
### 3. SSD-MobileNet v1

```baash 
$ python3 test_ssd_mobilenet_v1.py 
```

<span id="ssdlite-mobilenet-v2"></span>
### 4. SSDLite-MobileNet v2 (tflite)

* download the [ssdlite-mobilenet-v2](https://drive.google.com/file/d/1Ha9yfjkweCatEo6UoZgZyHMeyIBGe5FO/view?usp=sharing) file and put it to model_data file

```baash 
$ python3 test_ssdlite_mobilenet_v2.py 
```

## Compare

* tiny-YOLOv2

![](/out/tiny_yolo_dog.jpg)


* YOLOv3

![](/out/yolov3_dog.jpg)

* SSD-MobileNet v1

![](/out/ssd_mobilenet_v1_dog.jpg)

* SSDLite-MobileNet v2 (tflite)
![](/out/ssdlite_mobilenet_v2_dog.jpg)

## Acknowledgments

* Thanks to [keras-yolo3](https://github.com/qqwweee/keras-yolo3) for yolov3-keras part.
* Thanks to [mobile-object-detector-with-tensorflow-lite](https://medium.com/datadriveninvestor/mobile-object-detector-with-tensorflow-lite-9e2c278922d0) for ssdlite-mobilenet-v2 part.
