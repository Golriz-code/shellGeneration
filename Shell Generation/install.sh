#!/usr/bin/env sh
HOME=`pwd`
PYTHONPATH="${PYTHONPATH}:path of site packages of python"
export PYTHONPATH

cd $HOME/extension/chamfer2D
python setup.py install --user
