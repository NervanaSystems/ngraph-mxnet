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
set -e

cd "$HOME/ng-mx"

cd python && pip install -e . && pip install pytest nose scipy==1.0.0 &&  cd ../

echo "Look for libcuda.so"

echo `ls /usr/local/cuda/lib64/stubs/`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs/:/usr/local/cuda-9.2/lib64/stubs/:/usr/local/nvidia/lib64/libcuda.so.1

echo " LD_LIBRARY_PATH ==== ${LD_LIBRARY_PATH}"

echo "Check if "

#sudo ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

## Unit tests test_operator.py 
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/gpu/test_operator_gpu.py --verbose --capture=no --junit-xml=result_test_operator_gpu.xml --junit-prefix=result_test_operator_gpu"
eval $cmd