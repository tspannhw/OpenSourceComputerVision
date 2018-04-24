import numpy
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
import sys
import os
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

import numpy
from time import gmtime, strftime
start = time.time()
cap = cv2.VideoCapture(1)

# http://gluon-crash-course.mxnet.io/predict.html
def transform(data):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std


net = models.resnet50_v2(pretrained=True)


url = 'http://data.mxnet.io/models/imagenet/synset.txt'
fname = download(url)
with open(fname, 'r') as f:
    text_labels = [' '.join(l.split()[1:]) for l in f]

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/\
Golden_Retriever_medium-to-light-coat.jpg/\
365px-Golden_Retriever_medium-to-light-coat.jpg'
fname = download(url)

ret, frame = cap.read()
filename = 'images/gluon_image_{0}_{1}.jpg'.format('img',strftime("%Y%m%d%H%M%S",gmtime()))
cv2.imwrite(filename, frame)

x = image.imread(filename)

x = image.resize_short(x, 256)
x, _ = image.center_crop(x, (224,224))

prob = net(transform(x)).softmax()
idx = prob.topk(k=5)[0]
for i in idx:
    i = int(i.asscalar())
    print('prob=%.5f, %s' % (
        prob[0,i].asscalar() * 100, text_labels[i]))
