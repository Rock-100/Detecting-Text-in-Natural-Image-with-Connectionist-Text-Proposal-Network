#!/bin/bash

set -x
set -e

NET=$1

LOG=${NET}"/logs/ctpn_"${NET}".txt.`date +'%Y-%m-%d_%H-%M-%S'`"
echo Logging output to "$LOG"

python tools/train_net.py ${NET} 2>&1 |tee "$LOG"
