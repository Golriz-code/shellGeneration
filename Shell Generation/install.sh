#!/usr/bin/env sh
HOME=`pwd`
PYTHONPATH="${PYTHONPATH}:/home/golriz/projects/def-guibault/golriz/mar/lib/python3.8/site-packages"
export PYTHONPATH
# Chamfer Distance
source /home/golriz/projects/def-guibault/golriz/mar/bin/activate
cd $HOME/extension/chamfer2D
python setup.py install --user
