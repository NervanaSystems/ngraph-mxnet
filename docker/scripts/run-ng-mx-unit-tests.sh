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

cd "$HOME/ng-mx"

cd python && sudo -E pip install -e . && cd ../

### tests/python/unittest/ 

## Unit tests test_operator.py 
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_operator.py --verbose --capture=no --junit-xml=result_test_operator.xml --junit-prefix=result_test_operator"
eval $cmd

## Unit tests test_attr.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_attr.py --verbose --capture=no --junit-xml=result_test_attr.xml --junit-prefix=result_test_attr"
eval $cmd

## Unit tests test_autograd.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_autograd.py --verbose --capture=no --junit-xml=result_test_autograd.xml --junit-prefix=result_test_autograd"
eval $cmd

## Unit tests test_contrib_autograd.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_contrib_autograd.py --verbose --capture=no --junit-xml=result_test_contrib_autograd.xml --junit-prefix=result_test_contrib_autograd"
eval $cmd

## Unit tests test_contrib_krprod.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_contrib_krprod.py --verbose --capture=no --junit-xml=result_test_contrib_krprod.xml --junit-prefix=result_test_contrib_krprod"
eval $cmd

## Unit tests test_contrib_operator.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_contrib_operator.py --verbose --capture=no --junit-xml=result_test_contrib_operator.xml --junit-prefix=result_test_contrib_operator"
eval $cmd

## Unit tests test_contrib_text.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_contrib_text.py --verbose --capture=no --junit-xml=result_test_contrib_text.xml --junit-prefix=result_test_contrib_text"
eval $cmd

## Unit tests test_engine.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_engine.py --verbose --capture=no --junit-xml=result_test_engine.xml --junit-prefix=result_test_engine"
eval $cmd

## Unit tests test_executor.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_executor.py --verbose --capture=no --junit-xml=result_test_executor.xml --junit-prefix=result_test_executor"
eval $cmd

## Unit tests test_gluon.py
cmd="export MXNET_NGRAPH_GLUON=1 ; OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon.py --verbose --capture=no --junit-xml=result_test_gluon.xml --junit-prefix=result_test_gluon"
eval $cmd

## Unit tests test_gluon_contrib.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon_contrib.py --verbose --capture=no --junit-xml=result_test_gluon_contrib.xml --junit-prefix=result_test_gluon_contrib"
eval $cmd

## Unit tests test_gluon_data.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon_data.py --verbose --capture=no --junit-xml=result_test_gluon_data.xml --junit-prefix=result_test_gluon_data"
eval $cmd

## Unit tests test_gluon_data_vision.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon_data_vision.py --verbose --capture=no --junit-xml=result_test_gluon_data_vision.xml --junit-prefix=result_test_gluon_data_vision"
eval $cmd

## Unit tests test_gluon_model_zoo.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon_model_zoo.py --verbose --capture=no --junit-xml=result_test_gluon_model_zoo.xml --junit-prefix=result_test_gluon_model_zoo"
eval $cmd

## Unit tests test_gluon_rnn.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_gluon_rnn.py --verbose --capture=no --junit-xml=result_test_gluon_rnn.xml --junit-prefix=result_test_gluon_rnn"
eval $cmd

## Unit tests test_image.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_image.py --verbose --capture=no --junit-xml=result_test_image.xml --junit-prefix=result_test_image"
eval $cmd

## Unit tests test_infer_shape.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_infer_shape.py --verbose --capture=no --junit-xml=result_test_infer_shape.xml --junit-prefix=result_test_infer_shape"
eval $cmd

## Unit tests test_init.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_init.py --verbose --capture=no --junit-xml=result_test_init.xml --junit-prefix=result_test_init"
eval $cmd

## Unit tests test_io.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_io.py --verbose --capture=no --junit-xml=result_test_io.xml --junit-prefix=result_test_io"
eval $cmd

## Unit tests test_kvstore.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_kvstore.py --verbose --capture=no --junit-xml=result_test_kvstore.xml --junit-prefix=result_test_kvstore"
eval $cmd

## Unit tests test_loss.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_loss.py --verbose --capture=no --junit-xml=result_test_loss.xml --junit-prefix=result_test_loss"
eval $cmd

## Unit tests test_metric.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_metric.py --verbose --capture=no --junit-xml=result_test_metric.xml --junit-prefix=result_test_metric"
eval $cmd

## Unit tests test_model_parallel.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_model_parallel.py --verbose --capture=no --junit-xml=result_test_model_parallel.xml --junit-prefix=result_test_model_parallel"
eval $cmd

## Unit tests test_module.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_module.py --verbose --capture=no --junit-xml=result_test_module.xml --junit-prefix=result_test_module"
eval $cmd

## Unit tests test_multi_device_exec.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_multi_device_exec.py --verbose --capture=no --junit-xml=result_test_multi_device_exec.xml --junit-prefix=result_test_multi_device_exec"
eval $cmd

## Unit tests test_ndarray.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_ndarray.py --verbose --capture=no --junit-xml=result_test_ndarray.xml --junit-prefix=result_test_ndarray"
eval $cmd

##Unit tests test_optimizer.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_optimizer.py --verbose --capture=no --junit-xml=result_test_optimizer.xml --junit-prefix=result_test_optimizer"
eval $cmd

##Unit tests test_profiler.py
cmd="OMP_NUM_THREADS=4 pytest -s -n tests/python/unittest/test_profiler.py --verbose --capture=no --junit-xml=result_test_profiler.xml --junit-prefix=result_test_profiler"
eval $cmd

##Unit tests test_random.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_random.py --verbose --capture=no --junit-xml=result_test_random.xml --junit-prefix=result_test_random"
eval $cmd

##Unit tests test_recordio.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_recordio.py --verbose --capture=no --junit-xml=result_test_recordio.xml --junit-prefix=result_test_recordio"
eval $cmd

##Unit tests test_rnn.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_rnn.py --verbose --capture=no --junit-xml=result_test_rnn.xml --junit-prefix=result_test_rnn"
eval $cmd

##Unit tests test_sparse_ndarray.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_sparse_ndarray.py --verbose --capture=no --junit-xml=result_test_sparse_ndarray.xml --junit-prefix=result_test_sparse_ndarray"
eval $cmd

##Unit tests test_sparse_operator.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_sparse_operator.py --verbose --capture=no --junit-xml=result_test_sparse_operator.xml --junit-prefix=result_test_sparse_operator"
eval $cmd

##Unit tests test_symbol.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_symbol.py --verbose --capture=no --junit-xml=result_test_symbol.xml --junit-prefix=result_test_symbol"
eval $cmd

##Unit tests test_viz.py
cmd="OMP_NUM_THREADS=4 pytest -s -n 2 tests/python/unittest/test_viz.py --verbose --capture=no --junit-xml=result_test_viz.xml --junit-prefix=result_test_viz"
eval $cmd

### tests/cpp
make test -j$(nproc) USE_NGRAPH=1
export LD_LIBRARY_PATH="$HOME/ng-mx/ngraph_dist/lib"
./build/tests/cpp/mxnet_unit_tests --gtest_output="xml:result_test_cpp.xml"
