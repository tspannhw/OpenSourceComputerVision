# 2017 load pictures and analyze
# https://github.com/tspannhw/mxnet_rpi/blob/master/analyze.py
import time
import sys
import datetime
import subprocess
import urllib2
import os
import datetime
import traceback
import math
import random, string
import base64
import json
import mxnet as mx
import inception_predict
import numpy as np
import cv2
import random, string
import socket
import psutil
from time import sleep
from string import Template
from time import gmtime, strftime

# Time
start = time.time()
currenttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
host = os.uname()[1]
cpu = psutil.cpu_percent(interval=1)
if 1==1:
    f = open('/sys/class/thermal/thermal_zone0/temp', 'r')
    l = f.readline()
    ctemp = 1.0 * float(l)/1000
usage = psutil.disk_usage("/")
mem = psutil.virtual_memory()
diskrootfree =  "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
mempercent = mem.percent
external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

def IP_address():
        try:
            s = socket.socket(socket_family, socket.SOCK_DGRAM)
            s.connect(external_IP_and_port)
            answer = s.getsockname()
            s.close()
            return answer[0] if answer else None
        except socket.error:
            return None
ipaddress = IP_address()

face_cascade_path = '/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))

scale_factor = 1.1
min_neighbors = 3
min_size = (30, 30)

cap = cv2.VideoCapture(0)
packet_size=3000

def randomword(length):
 return ''.join(random.choice(string.lowercase) for i in range(length))

#while True:

# Create unique image name
uniqueid = 'mxnet_uuid_{0}_{1}'.format(randomword(3),strftime("%Y%m%d%H%M%S",gmtime()))

ret, frame = cap.read()

imgdir = 'images/'
filename = 'tx1_image_{0}_{1}.jpg'.format(randomword(3),strftime("%Y%m%d%H%M%S",gmtime()))
cv2.imwrite(imgdir + filename, frame)

# Run inception prediction on image
try:
     topn = inception_predict.predict_from_local_file(imgdir + filename, N=5)
except:
     errorcondition = "true"

# CPU Temp
f = open("/sys/devices/virtual/thermal/thermal_zone1/temp","r")
cputemp = str( f.readline() )
cputemp = cputemp.replace('\n','')
cputemp = cputemp.strip()
cputemp = str(round(float(cputemp)) / 1000)
cputempf = str(round(9.0/5.0 * float(cputemp) + 32))
f.close()

# GPU Temp
f = open("/sys/devices/virtual/thermal/thermal_zone2/temp","r")
gputemp = str( f.readline() )
gputemp = gputemp.replace('\n','')
gputemp = gputemp.strip()
gputemp = str(round(float(gputemp)) / 1000)
gputempf = str(round(9.0/5.0 * float(gputemp) + 32))
f.close()

# NVidia Face Detect
p = os.popen('/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/jetson-inference/build/aarch64/bin/facedetect.sh ' + filename).read()
face = p.replace('\n','|')
face = face.strip()

# NVidia Image Net Classify
p2 = os.popen('/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/jetson-inference/build/aarch64/bin/runclassify.sh ' + filename).read()
imagenet = p2.replace('\n','|')
imagenet = imagenet.strip()

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

# OpenCV

infname = "/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/images" + filename
flags = cv2.CASCADE_SCALE_IMAGE
#image_path = os.path.expanduser(infname)
image = cv2.imread(imgdir + filename)
#frame
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size, flags = flags)

# Create Face Images

x = 0
y = 0
w = 0
h = 0
outfilename = filename
outfname = filename
cvface = ''
cvfilename = ''

for( x1, y1, w1, h1 ) in faces:
 cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
 outfname = "/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/images/%s.faces.jpg" % os.path.basename(infname)
 cv2.imwrite(os.path.expanduser(outfname), image)
 cvfilename += outfname
 cvface += 'Face {0}'.format(faces)
 outfilename = outfname
 x = x1
 y = y1
 w = w1
 h = h1

endtime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
end = time.time()
row = { 'uuid': uniqueid,  'top1pct': top1pct, 'top1': top1, 'top2pct': top2pct, 'top2': top2,'top3pct': top3pct, 'top3': top3,'top4pct': top4pct,'top4': top4, 'top5pct': top5pct,'top5': top5, 'gputemp': gputemp, 'imagefilename': filename, 'gputempf': gputempf, 'cputempf': cputempf, 'runtime': str(round(end - start)), 'facedetect': face, 'imagenet': imagenet, 'ts': currenttime, 'endtime': endtime, 'host': host, 'memory': mempercent, 'diskfree': diskrootfree, 'cputemp': round(ctemp,2), 'ipaddress': ipaddress, 'x': str(x), 'y': str(y), 'w': str(w), 'h': str(h), 'filename': outfname, 'cvface': cvface, 'cvfilename': cvfilename }

json_string = json.dumps(row)

print (json_string )
