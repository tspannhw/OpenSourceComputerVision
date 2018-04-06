# Forked From https://github.com/minimaxir/person-blocker
# License
#MIT
#
#Code used from Mask R-CNN by Matterport, Inc. (MIT-Licensed), with minor alterations and copyright notices retained.

import os
import sys
import argparse
import numpy as np
import coco
import utils
import model as modellib
from classes import get_class_names, InferenceConfig
from ast import literal_eval as make_tuple
import imageio
import visualize
import os.path
import re
import datetime
import math
import random, string
import base64
import json
import socket
import psutil
import subprocess
import time
import uuid
import cv2
import math
import random, string
import time
from time import gmtime, strftime

cap = cv2.VideoCapture(1)
packet_size=3000
ret, frame = cap.read()
filename = 'images2/tx1_image_{0}_{1}.jpg'.format(uuid.uuid4(),strftime("%Y%m%d%H%M%S",gmtime()))
cv2.imwrite(filename, frame)

from time import sleep
from time import gmtime, strftime
start = time.time()

# minor fork by tim spann for nifi usage
# 2018-april-4
# Creates a color layer and adds Gaussian noise.
# For each pixel, the same noise value is added to each channel
# to mitigate hue shfting.

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

def create_noisy_color(image, color):
    color_mask = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=color)

    noise = np.random.normal(0, 25, (image.shape[0], image.shape[1]))
    noise = np.repeat(np.expand_dims(noise, axis=2), repeats=3, axis=2)
    mask_noise = np.clip(color_mask + noise, 0., 255.)
    return mask_noise


# Helper function to allow both RGB triplet + hex CL input

def string_to_rgb_triplet(triplet):

    if '#' in triplet:
        # http://stackoverflow.com/a/4296727
        triplet = triplet.lstrip('#')
        _NUMERALS = '0123456789abcdefABCDEF'
        _HEXDEC = {v: int(v, 16)
                   for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
        return (_HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]],
                _HEXDEC[triplet[4:6]])

    else:
        # https://stackoverflow.com/a/9763133
        triplet = make_tuple(triplet)
        return triplet


def person_blocker(args):

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = args.model or os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # Required to load model

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load model and config
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = imageio.imread(filename)

    # Create masks for all objects
    results = model.detect([image], verbose=0)
    r = results[0]

    if args.labeled:
        position_ids = ['[{}]'.format(x)
                        for x in range(r['class_ids'].shape[0])]
        visualize.display_instances(image, r['rois'],
                                    r['masks'], r['class_ids'],
                                    get_class_names(), position_ids)
        sys.exit()

    # Filter masks to only the selected objects
    objects = np.array(args.objects)

    # Object IDs:
    if np.all(np.chararray.isnumeric(objects)):
        object_indices = objects.astype(int)
    # Types of objects:
    else:
        selected_class_ids = np.flatnonzero(np.in1d(get_class_names(),
                                                    objects))
        object_indices = np.flatnonzero(
            np.in1d(r['class_ids'], selected_class_ids))

    mask_selected = np.sum(r['masks'][:, :, object_indices], axis=2)

    # Replace object masks with noise
    mask_color = string_to_rgb_triplet(args.color)
    image_masked = image.copy()
    noisy_color = create_noisy_color(image, mask_color)
    image_masked[mask_selected > 0] = noisy_color[mask_selected > 0]

    img_dir = '/Volumes/seagate/projects/person-blocker/images2/'
    img_pre = 'person_blocked_{0}'.format(strftime("%Y%m%d%H%M%S",gmtime()))
    img_name = img_dir + img_pre + '.png'
    gif_name = img_dir + img_pre + '.gif'

    imageio.imwrite(img_name, image_masked)

    # Create GIF. The noise will be random for each frame,
    # which creates a "static" effect
    # this works great, but takes some time and produces 7+ meg file
    #images = [image_masked]
    #num_images = 10   # should be a divisor of 30
    #
    #for _ in range(num_images - 1):
    #    new_image = image.copy()
    #    noisy_color = create_noisy_color(image, mask_color)
    #    new_image[mask_selected > 0] = noisy_color[mask_selected > 0]
    #    images.append(new_image)
    #
    #imageio.mimsave(gif_name, images, fps=30., subrectangles=True)

    # print json
    try:
      # Create unique image name
      uniqueid = 'person_uuid_{0}_{1}'.format(strftime("%Y%m%d%H%M%S%f",gmtime()),uuid.uuid4())
      host = os.uname()[1]
      currenttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
      ipaddress = IP_address()
      end = time.time()
      row = { 'uuid': uniqueid, 'runtime': str(round(end - start)), 'host': host, 'ts': currenttime, 'ipaddress': ipaddress, 'imagefilename': img_pre, 'originalfilename': filename }
      print( json.dumps(row) )

    except:
      print("{\"message\": \"Failed to run\"}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Blocker - Automatically "block" people '
                    'in images using a neural network.')
    parser.add_argument('-i', '--image',  help='Image file name.',
                        required=False)
    parser.add_argument(
        '-m', '--model',  help='path to COCO model', default=None)
    parser.add_argument('-o',
                        '--objects', nargs='+',
                        help='object(s)/object ID(s) to block. ' +
                        'Use the -names flag to print a list of ' +
                        'valid objects',
                        default='person')
    parser.add_argument('-c',
                        '--color', nargs='?', default='(255, 255, 255)',
                        help='color of the "block"')
    parser.add_argument('-l',
                        '--labeled', dest='labeled',
                        action='store_true',
                        help='generate labeled image instead')
    parser.add_argument('-n',
                        '--names', dest='names',
                        action='store_true',
                        help='prints class names and exits.')
    parser.set_defaults(labeled=False, names=False)
    args = parser.parse_args()

    if args.names:
        print(get_class_names())
        sys.exit()

    person_blocker(args)
