#!/usr/bin/bash

yank analyze report --yaml sams.yaml --skipunbiasing --report --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/sams --serial=sams.pkl --fulltraj
yank analyze report --yaml repex.yaml --skipunbiasing --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/repex --serial=repex.pkl --fulltraj
