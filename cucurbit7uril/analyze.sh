#!/usr/bin/bash

yank analyze --yaml sams.yaml --skipunbiasing --serial=sams.pkl --fulltraj
yank analyze --yaml repex.yaml --skipunbiasing --serial=repex.pkl --fulltraj
