#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# ****************************************************************************
# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

# Forked by Tim Spann and added Sense Hat Code and JSON
# 2017-December-28

from sense_hat import SenseHat
import json
import sys, socket
import os
import psutil
import subprocess
import time
import datetime
from time import sleep
from time import gmtime, strftime
import mvnc.mvncapi as mvnc
import skimage
from skimage import io, transform
import numpy
import json
import traceback
import math
import random, string
import base64
import json
import mxnet as mx
import numpy as np
import time
import cv2, os, urllib
from collections import namedtuple

start = time.time()
starttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())

# User modifiable input parameters
NCAPPZOO_PATH           = os.path.expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/graph'
IMAGE_PATH              = sys.argv[1]
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEAN              = [ 104.00698793, 116.66876762, 122.67891434]
IMAGE_STDDEV            = 1
IMAGE_DIM               = ( 224, 224 )

Batch = namedtuple('Batch', ['data'])

# Load the symbols for the networks
with open('/opt/demo/incubator-mxnet/synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('/opt/demo/incubator-mxnet/Inception-BN', 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)

'''
Function to predict objects by giving the model a pointer to an image file and running a forward pass through the model.
inputs:
filename = jpeg file of image to classify objects in
mod = the module object representing the loaded model
synsets = the list of symbols representing the model
N = Optional parameter denoting how many predictions to return (default is top 5)
outputs:
python list of top N predicted objects and corresponding probabilities
'''
def predict(filename, mod, synsets, N=5):
    tic = time.time()
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    toc = time.time()
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    topN = []
    a = np.argsort(prob)[::-1]
    for i in a[0:N]:
        topN.append((prob[i], synsets[i]))
    return topN


# Code to download an image from the internet and run a prediction on it
def predict_from_url(url, N=5):
    filename = url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print( "Failed to download" )
    else:
        return predict(filename, mod, synsets, N)

# Code to predict on a local file
def predict_from_local_file(filename, N=5):
    return predict(filename, mod, synsets, N)

packet_size=3000

# Create unique image name
uniqueid = 'mxnet_uuid_{0}_{1}.json'.format('json',strftime("%Y%m%d%H%M%S",gmtime()))

filename = IMAGE_PATH
topn = []
# Run inception prediction on image
try:
     topn = predict_from_local_file(filename, N=5)
except:
     print("Error")
     errorcondition = "true"

# ---- Step 1: Open the enumerated device and get a handle to it -------------

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len( devices ) == 0:
    print( 'No devices found' )
    quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device( devices[0] )
device.OpenDevice()

# ---- Step 2: Load a graph file onto the NCS device -------------------------

# Read the graph file into a buffer
with open( GRAPH_PATH, mode='rb' ) as f:
    blob = f.read()

# Load the graph buffer into the NCS
graph = device.AllocateGraph( blob )

# ---- Step 3: Offload image onto the NCS to run inference -------------------

# Read & resize image [Image size is defined during training]
img = print_img = skimage.io.imread( IMAGE_PATH )
img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]

# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype( numpy.float32 )
img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

# Load the image as a half-precision floating point array
graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )

# ---- Step 4: Read & print inference results from the NCS -------------------

# Get the results from NCS
output, userobj = graph.GetResult()

labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )

order = output.argsort()[::-1][:6]

#### Initialization

external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

host = os.uname()[1]

def getCPUtemperature():
    res = os.popen('vcgencmd measure_temp').readline()
    return(res.replace("temp=","").replace("'C\n",""))

def IP_address():
        try:
            s = socket.socket(socket_family, socket.SOCK_DGRAM)
            s.connect(external_IP_and_port)
            answer = s.getsockname()
            s.close()
            return answer[0] if answer else None
        except socket.error:
            return None

cpuTemp=int(float(getCPUtemperature()))
ipaddress = IP_address()

host = os.uname()[1]
rasp = ('armv' in os.uname()[4])
cpu = psutil.cpu_percent(interval=1)
if rasp:
    f = open('/sys/class/thermal/thermal_zone0/temp', 'r')
    l = f.readline()
    ctemp = 1.0 * float(l)/1000
usage = psutil.disk_usage("/")
mem = psutil.virtual_memory()
diskrootfree =  "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
mempercent = mem.percent
external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

ipaddress = IP_address()

# Sense Hat
sense = SenseHat()
sense.clear()
temp = sense.get_temperature()
temp = round(temp, 2)
humidity = sense.get_humidity()
humidity = round(humidity, 1)
pressure = sense.get_pressure()
pressure = round(pressure, 1)
orientation = sense.get_orientation()
pitch = orientation['pitch']
roll = orientation['roll']
yaw = orientation['yaw']
acceleration = sense.get_accelerometer_raw()
x = acceleration['x']
y = acceleration['y']
z = acceleration['z']
#cputemp = out
x=round(x, 0)
y=round(y, 0)
z=round(z, 0)
pitch=round(pitch,0)
roll=round(roll,0)
yaw=round(yaw,0)

try:
     # 5 MXNET Analysis
     top1 = str(topn[0][1])
     top1pct = str(round(topn[0][0],3) * 100)

     top2 = str(topn[1][1])
     top2pct = str(round(topn[1][0],3) * 100)

     top3 = str(topn[2][1])
     top3pct = str(round(topn[2][0],3) * 100)

     top4 = str(topn[3][1])
     top4pct = str(round(topn[3][0],3) * 100)

     top5 = str(topn[4][1])
     top5pct = str(round(topn[4][0],3) * 100)

     end = time.time()
     currenttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())

     row = { 'uuid': uniqueid,  'top1pct': top1pct, 'top1': top1, 'top2pct': top2pct, 'top2': top2,'top3pct': top3pct, 'top3': top3,'top4pct': top4pct,'top4': top4, 'top5pct': top5pct,'top5': top5, 'imagefilename': filename, 'runtime': str(round(end - start)),
             'cputemp2': round(ctemp,2), 'temp': temp, 'tempf': round(((temp * 1.8) + 12),2), 'humidity': humidity, 'pressure': pressure, 'pitch': pitch, 'roll': roll, 'yaw': yaw, 'x': x, 'y': y, 'z': z,'memory': mempercent, 'diskfree': diskrootfree, 'label1': labels[order[0]],
             'label2': labels[order[1]], 'label3': labels[order[2]], 'label4': labels[order[3]], 'label5': labels[order[4]], 'currenttime': currenttime, 'host': host, 'cputemp': round(cpuTemp,2), 'ipaddress': ipaddress, 'starttime': starttime }

     json_string = json.dumps(row)
     print (json_string)

except:
     print("{\"message\": \"Failed to run\"}")


# ---- Step 5: Unload the graph and close the device -------------------------

graph.DeallocateGraph()
device.CloseDevice()

# ==== End of file ===========================================================
