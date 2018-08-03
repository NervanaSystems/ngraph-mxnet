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
#set -e

cd "$HOME/ng-mx"

cd python && sudo -E pip install -e . && cd ../

## Unit tests test_operator.py 
#cmd="OMP_NUM_THREADS=4 pytest -s tests/python/unittest/test_operator.py --verbose --capture=no --junit-xml=result_test_operator.xml --junit-prefix=result_test_operator"
cmd="OMP_NUM_THREADS=4 pytest -s tests/python/unittest/test_contrib_krprod.py --verbose --capture=no --junit-xml=result_test_contrib_krprod.xml --junit-prefix=result_test_contrib_krprod"
eval $cmd
