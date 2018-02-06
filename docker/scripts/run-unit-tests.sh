#!/bin/bash
#Author:  Lam Nguyen
set -u
set -e

cd "$HOME/ng-mx"

cd python && sudo -E pip install -e . && cd ../

## Unit tests test_operator.py 
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_operator.py --verbose --capture=no"
eval $cmd