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

cd python && pip install -e . && pip install pytest nose scipy &&  cd ../

## 1. Unit tests test_device.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_device.py --verbose --capture=no --junit-xml=result_test_device_gpu.xml --junit-prefix=result_test_device_gpu"
eval $cmd

## 2. Unit tests test_operator_gpu.py will be failed with CUDA9 (NGRAPH-3118)
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_operator_gpu.py --verbose --capture=no --junit-xml=result_test_operator_gpu.xml --junit-prefix=result_test_operator_gpu"
eval $cmd

## 3. Unit tests test_forward.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_forward.py --verbose --capture=no --junit-xml=result_test_forward_gpu.xml --junit-prefix=result_test_forward_gpu"
eval $cmd

## 4. Unit tests test_gluon_gpu.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_gluon_gpu.py --verbose --capture=no --junit-xml=result_test_gluon_gpu.xml --junit-prefix=result_test_gluon_gpu"
eval $cmd

## 5. Unit tests test_gluon_model_zoo_gpu.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_gluon_model_zoo_gpu.py --verbose --capture=no --junit-xml=result_test_gluon_model_zoo_gpu.xml --junit-prefix=result_test_gluon_model_zoo_gpu"
eval $cmd

## 6. Unit tests test_kvstore_gpu.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_kvstore_gpu.py --verbose --capture=no --junit-xml=result_test_kvstore_gpu.xml --junit-prefix=result_test_kvstore_gpu"
eval $cmd

## 7. Unit tests test_nccl.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_nccl.py --verbose --capture=no --junit-xml=result_test_nccl_gpu.xml --junit-prefix=result_test_nccl_gpu"
eval $cmd

## 8. Unit tests test_rtc.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_rtc.py --verbose --capture=no --junit-xml=result_test_rtc_gpu.xml --junit-prefix=result_test_rtc_gpu"
eval $cmd

## 9. Unit tests test_tvm_brigde.py
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_tvm_brigde.py --verbose --capture=no --junit-xml=result_test_tvm_brigde_gpu.xml --junit-prefix=result_test_tvm_brigde_gpu"
eval $cmd
