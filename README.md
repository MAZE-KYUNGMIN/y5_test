# y5_test

```

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
```
