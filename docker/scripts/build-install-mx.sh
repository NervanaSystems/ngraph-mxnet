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

echo "****************************************************************************************"
echo "Build and install Ngraph_MXnet..."
echo "****************************************************************************************"

cd "${MX_DIR}"
git submodule update --init --recursive
make USE_NGRAPH=1 USE_GPERFTOOLS=0 USE_JEMALLOC=0  USE_CUDA=0 DEBUG=0 -j $(nproc)
export LD_LIBRARY_PATH=${MX_DIR}/3rdparty/ngraph/install/lib:${LD_LIBRARY_PATH}

if [ ! -f "$LD_LIBRARY_PATH/libngraph.so" ] ; then
	( >&2 echo "FATAL ERROR: Can not found libngraph.so. Exiting ...." )
  	exit 1
else
	echo "Success to install 3rdparty Ngraph."
fi	

if [ ! -f "$LD_LIBRARY_PATH/libmxnet.so" ] ; then
  ( >&2 echo "FATAL ERROR: Can not found libmxnet.so. Exiting ...." )
  exit 1
else
   echo "Success to install ngraph-mxnet."
fi

if [ ! -f "$LD_LIBRARY_PATH/libmkldnn.so" ] ; then
  ( >&2 echo "FATAL ERROR: libmkldnn.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
  exit 1
fi
