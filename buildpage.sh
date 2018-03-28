#!/bin/sh

echo '<html><head><meta http-equiv="refresh" content="60"> <title>NiFi List Images</title> </head> <body> <p> <br> <b>List Images</b> <br><br>'
sed 's/^.*/<a target="_new" href="http:\/\/princeton1\.field\.hortonworks\.com\:9099\?img_name\=&">&<\/a><br\/>/'
echo '</body></html>'
