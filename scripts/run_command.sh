#!/usr/bin/env/bash

line=$1
log=$2
set -x
eval $line
echo "DONE!!! $line" >> $log
