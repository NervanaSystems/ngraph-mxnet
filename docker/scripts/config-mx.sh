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

declare SCRIPT_NAME="$(basename "${0}")"
declare THIS_SCRIPT_DIR="$( cd $(dirname "${BASH_SOURCE[0]}") && pwd )"
declare MX_DIR="$(cd "${THIS_SCRIPT_DIR}/../.." && pwd)"
declare NGRAPH_DIR="${MX_DIR}/ngraph_dist"
echo "Configure Mxnet..."

#Modify the config.mk 
#USE_NGRAPH = 1
#USE_MKL2017 = 0
#USE_NNPACK = 0
#Fill out the directory of NGRAPH_DIR

sed -e '/^USE_NGRAPH/s/.*/USE_NGRAPH = 1/' -e '/^USE_MKL2017/s/.*/USE_MKL2017 = 0/' -e '/^USE_NNPACK/s/.*/USE_NNPACK = 0/' -e "s@\(NGRAPH_DIR = *\)@\1 ${NGRAPH_DIR}@g" ${MX_DIR}/make/config.mk > ${MX_DIR}/make/config.mk.tmp

cp ${MX_DIR}/make/config.mk.tmp  ${MX_DIR}/make/config.mk

echo "Success."