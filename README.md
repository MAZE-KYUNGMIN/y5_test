# y5_test

```
https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi ##sample

## blog
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git 



mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/ tflite1


~ $ cd tflite1
~/tflite1 $ sudo pip3 install virtualenv
~/tflite1 $ python3 -m venv tflite1-env
~/tflite1 $ source tflite1-env/bin/activate
(tflite1-env)~/tflite1 $ bash get_pi_requirements.sh


~ $ cd tflite1
~/tflite1 $ source tflite1-env/bin/activate
(tflite1-env)~/tflite1 $


(tflite1-env)~/tflite1 $ unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d TFLite_model


(tflite1-env)~/tflite1 $ python3 TFLite_detection_webcam.py --modeldir=TFLite_model


git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git



mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/ tflite1


~ $ cd tflite1
~/tflite1 $ sudo pip3 install virtualenv
~/tflite1 $ python3 -m venv tflite1-env
~/tflite1 $ source tflite1-env/bin/activate
(tflite1-env)~/tflite1 $ bash get_pi_requirements.sh


~ $ cd tflite1
~/tflite1 $ source tflite1-env/bin/activate
(tflite1-env)~/tflite1 $


(tflite1-env)~/tflite1 $ unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d TFLite_model


(tflite1-env)~/tflite1 $ python3 TFLite_detection_webcam.py --modeldir=TFLite_model

```

```

from cv2 import VideoCapture
from tflite_support.task import vision
import cv2

# detector = vision.ObjectDetector.create_from_file('yolo_v5_tflite.tflite')

# cap = VideoCapture(0)

# while cap.isOpened():

#     _, frame = cap.read()
#     image = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
#     tensor_image = vision.TensorImage.create_from_array(image)
#     detector.detect(tensor_image)

#     # cv2.imshow('object detect', image)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



pi@raspberrypi:~/examples/lite/examples/object_detection/raspberry_pi $ python detect.py --model yolov5_lite.tflite
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Traceback (most recent call last):
  File "/home/pi/examples/lite/examples/object_detection/raspberry_pi/detect.py", line 150, in <module>
    main()
  File "/home/pi/examples/lite/examples/object_detection/raspberry_pi/detect.py", line 145, in main
    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
  File "/home/pi/examples/lite/examples/object_detection/raspberry_pi/detect.py", line 63, in run
    detector = vision.ObjectDetector.create_from_options(options)
  File "/home/pi/.local/lib/python3.9/site-packages/tensorflow_lite_support/python/task/vision/object_detector.py", line 82, in create_from_options
    detector = _CppObjectDetector.create_from_options(options.base_options,
RuntimeError: Input tensor has type kTfLiteFloat32: it requires specifying NormalizationOptions metadata to preprocess input images.

```
