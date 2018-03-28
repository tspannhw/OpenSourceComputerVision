import cv2
import os
import sys
import json
import socket
import psutil
import subprocess
import time
import datetime
from time import sleep
from time import gmtime, strftime
from string import Template
# forked from https://gist.github.com/dannguyen/cfa2fb49b28c82a1068f
# first argument is the haarcascades path

currenttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
host = os.uname()[1]
#print(os.uname())
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
#p = subprocess.Popen(['/opt/vc/bin/vcgencmd','measure_temp'], stdout=subprocess.PIPE,
#    stderr=subprocess.PIPE)
#out, err = p.communicate()
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
flags = cv2.CASCADE_SCALE_IMAGE

print('[')
for infname in sys.argv[1:]:
   image_path = os.path.expanduser(infname)
   image = cv2.imread(image_path)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size, flags = flags)
   print('Faces: {0}'.format(len(faces)))
   print('Face {0}'.format(faces))
   for( x, y, w, h ) in faces:
     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
     outfname = "/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/images/%s.faces.jpg" % os.path.basename(infname)
     cv2.imwrite(os.path.expanduser(outfname), image)
     endtime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
     # row =  { 'ts': currenttime, 'endtime': endtime, 'host': host, 'memory': mempercent, 'diskfree': diskrootfree, 'cputemp': round(ctemp,2), 'ipaddress': ipaddress, 'x': x, 'y': y, 'w': w, 'h': h, 'filename': outfname }
     row =  { 'ts': currenttime, 'endtime': endtime, 'host': host, 'memory': mempercent, 'diskfree': diskrootfree, 'cputemp': round(ctemp,2), 'ipaddress': ipaddress, 'x': str(x), 'y': str(y), 'w': str(w), 'h': str(h), 'filename': outfname }

     json_string = json.dumps(row)
     print(json_string)
print(']')
