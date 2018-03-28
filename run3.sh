#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

fswebcam -q -r 1280x720 --no-banner /opt/demo/images/$DATE.jpg

python2 -W ignore /opt/demo/classify_image.py /opt/demo/images/$DATE.jpg 2>/dev/null
