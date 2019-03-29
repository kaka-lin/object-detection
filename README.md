# Object Detection
This is exercise for object detection with ```ssd_mobilenet``` and ```tiny-yolo(coco)```

Add: ```YOLOV3```


## Usage

### 1. tiny-yolo

* download the [tiny-yolo](https://drive.google.com/file/d/14-5ZojD1HSgMKnv6_E3WUcBPxaVm52X2/view?usp=sharing) file and put it to model_data file

```baash 
$ python3 test_tiny_yolo.py 
```

### 2. ssd-mobilenet

```baash 
$ python3 test_ssd_mobilenet.py 
```

### 3. YOLOV3

* download the [yolov3](https://drive.google.com/open?id=1vdD9TPiTWqvPxtCXdbVSKKksSdu0j_Hn) file and put it to model_data file

```baash 
$ python3 test_yolov3.py 
```

## YOLOV2 vs YOLOV3

* YOLOV2

![YOLOV2](/out/dog2.jpg)


* YOLOV3

![YOLOV2](/out/dog.jpg)

## Acknowledgments

* Thanks to [keras-yolo3](https://github.com/qqwweee/keras-yolo3) for yolov3-keras part
