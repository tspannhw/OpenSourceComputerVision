/Volumes/seagate/IoTFusion/minifi-toolkit-0.4.0/bin/config.sh transform $1 config.yml
scp config.yml pi@192.168.1.156:/opt/demo/minifi-0.4.0/conf/
