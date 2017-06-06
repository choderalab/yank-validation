#!/bin/bash

#
# Subset of the FreeSolv Database Hydration Free Energy run script (serial mode)
#

# Run the simulation
echo "Running simulation..."
yank script --yaml=yank.yaml
