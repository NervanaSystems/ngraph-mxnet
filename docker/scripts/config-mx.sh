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


set -u
set -e

declare SCRIPT_NAME="$(basename "${0}")"
declare THIS_SCRIPT_DIR="$( cd $(dirname "${BASH_SOURCE[0]}") && pwd )"
declare MX_DIR="$(cd "${THIS_SCRIPT_DIR}/../.." && pwd)"
declare WARPCTC_DIR="${MX_DIR}/warp-ctc"
cd ${WARPCTC_DIR}
echo "Installing WARP-CTC..."
mkdir build
cd build
cmake ../
make  -j 

rc=$?
if [ $rc -ne 0 ];then
    echo "Failed to build warp-ctc"
    exit $rc
fi

cd ${MX_DIR}
#sed -e '/^USE_NGRAPH/s/.*/USE_NGRAPH = 1/' -e '/^USE_MKL2017/s/.*/USE_MKL2017 = 0/' -e '/^USE_NNPACK/s/.*/USE_NNPACK = 0/' -e "s@\(NGRAPH_DIR = *\)@\1 ${NGRAPH_DIR}@g" ${MX_DIR}/make/config.mk > ${MX_DIR}/make/config.mk.tmp
echo "Updatind config.mk to enable WARP-CTC..."
echo "WARPCTC_PATH = ${WARPCTC_DIR}" >> ${MX_DIR}/make/config.mk 
echo "MXNET_PLUGINS += plugin/warpctc/warpctc.mk" >> ${MX_DIR}/make/config.mk 

export LD_LIBRARY_PATH="${WARPCTC_DIR}/build"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "Success."
