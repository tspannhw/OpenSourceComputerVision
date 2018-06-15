# Based on https://gluon-cv.mxnet.io/build/examples_detection/demo_ssd.html#sphx-glr-build-examples-detection-demo-ssd-py
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy
import base64
import uuid
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
from mxnet import image
import time
import sys
import datetime
import subprocess
import os
from PIL import Image
import datetime
import traceback
import math
import random, string
import base64
import json
from time import gmtime, strftime
import mxnet as mx
import inception_predict
import numpy as np
import cv2
import math
import random, string
import time
import numpy
import random, string
import time
import psutil
import paho.mqtt.client as mqtt
import scipy.misc
from time import gmtime, strftime
start = time.time()
cap = cv2.VideoCapture(1)   # 0 - laptop   #1 - monitor
ret, frame = cap.read()
uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename = 'images/gluoncv_image_{0}.jpg'.format(uuid)
filename2 = 'images/gluoncv_image_processed_{0}.jpg'.format(uuid)
cv2.imwrite(filename, frame)

# model zoo for SSD 512 RESNET 50 v1 VOC
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

#im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                          'gluoncv/detection/street_small.jpg?raw=true',
#                          path='street_small.jpg')

x, img = data.transforms.presets.ssd.load_test(filename, short=512)

end = time.time()
row = { }
row['imgname'] = filename
row['host'] = os.uname()[1]
row['shape'] = str(x.shape)
row['end'] = '{0}'.format( str(end ))
row['te'] = '{0}'.format(str(end-start))
row['battery'] = psutil.sensors_battery()[0]
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['cpu'] = psutil.cpu_percent(interval=1)
usage = psutil.disk_usage("/")
row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
row['memory'] = psutil.virtual_memory().percent
row['id'] = str(uuid)
json_string = json.dumps(row)
# print(json_string)

# MQTT
client = mqtt.Client()
client.username_pw_set("user","pass")
client.connect("server", 17769, 60)
client.publish("gluoncv", payload=json_string, qos=0, retain=True)

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)

plt.savefig(filename2)
# plt.show()
