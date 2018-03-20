# *******************************************************************************
# * Copyright 2018 Intel Corporation
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# ********************************************************************************

#!/bin/bash
#Author:  Lam Nguyen
set -u
##set -e

cd "$HOME/ng-mx"

cd python && sudo -E pip install -e . && cd ../

### tests/python/unittest/ 

## Unit tests test_operator.py 
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_operator.py --verbose --capture=no"
eval $cmd

## Unit tests test_attr.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_attr.py --verbose --capture=no"
eval $cmd

## Unit tests test_autograd.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_autograd.py --verbose --capture=no"
eval $cmd

## Unit tests test_contrib_autograd.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_contrib_autograd.py --verbose --capture=no"
eval $cmd

## Unit tests test_contrib_krprod.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_contrib_krprod.py --verbose --capture=no"
eval $cmd

## Unit tests test_contrib_operator.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_contrib_operator.py --verbose --capture=no"
eval $cmd

## Unit tests test_contrib_text.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_contrib_text.py --verbose --capture=no"
eval $cmd

## Unit tests test_engine.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_engine.py --verbose --capture=no"
eval $cmd

## Unit tests test_executor.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_executor.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon_contrib.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon_contrib.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon_data.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon_data.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon_data_vision.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon_data_vision.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon_model_zoo.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon_model_zoo.py --verbose --capture=no"
eval $cmd

## Unit tests test_gluon_rnn.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_gluon_rnn.py --verbose --capture=no"
eval $cmd

## Unit tests test_image.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_image.py --verbose --capture=no"
eval $cmd

## Unit tests test_infer_shape.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_infer_shape.py --verbose --capture=no"
eval $cmd

## Unit tests test_init.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_init.py --verbose --capture=no"
eval $cmd

## Unit tests test_io.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_io.py --verbose --capture=no"
eval $cmd

## Unit tests test_kvstore.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_kvstore.py --verbose --capture=no"
eval $cmd

## Unit tests test_loss.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_loss.py --verbose --capture=no"
eval $cmd

## Unit tests test_metric.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_metric.py --verbose --capture=no"
eval $cmd

## Unit tests test_model_parallel.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_model_parallel.py --verbose --capture=no"
eval $cmd

## Unit tests test_module.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_module.py --verbose --capture=no"
eval $cmd

## Unit tests test_multi_device_exec.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_multi_device_exec.py --verbose --capture=no"
eval $cmd

## Unit tests test_ndarray.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_ndarray.py --verbose --capture=no"
eval $cmd

##Unit tests test_optimizer.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_optimizer.py --verbose --capture=no"
eval $cmd

##Unit tests test_profiler.py
cmd="OMP_NUM_THREADS=4 pytest tests/python/unittest/test_profiler.py--verbose --capture=no"
eval $cmd

##Unit tests test_random.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_random.py --verbose --capture=no"
eval $cmd

##Unit tests test_recordio.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_recordio.py --verbose --capture=no"
eval $cmd

##Unit tests test_rnn.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_rnn.py --verbose --capture=no"
eval $cmd

##Unit tests test_sparse_ndarray.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_sparse_ndarray.py --verbose --capture=no"
eval $cmd

##Unit tests test_sparse_operator.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_sparse_operator.py --verbose --capture=no"
eval $cmd

##Unit tests test_symbol.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_symbol.py --verbose --capture=no"
eval $cmd

##Unit tests test_viz.py
cmd="OMP_NUM_THREADS=4 pytest -n 2 tests/python/unittest/test_viz.py --verbose --capture=no"
eval $cmd
