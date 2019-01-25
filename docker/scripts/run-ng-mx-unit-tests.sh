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

cd python && pip install -e . && cd ../

pip install pytest
pip install nose
pip install mock
pip install scipy==1.0.0

### tests/python/unittest/ 
cmd="OMP_NUM_THREADS=4 $(which python) -m pytest -s tests/python/unittest --verbose --capture=no --junit-xml=result_unittest.xml --junit-prefix=result_test_unittestr"
eval $cmd


### tests/cpp
#make test -j$(nproc) USE_NGRAPH=1
#./build/tests/cpp/mxnet_unit_tests --gtest_output="xml:result_test_cpp.xml"
