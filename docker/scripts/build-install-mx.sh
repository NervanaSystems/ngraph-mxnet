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
declare NGRAPH_INSTALL_DIR="${HOME}/ngraph_dist"
declare MX_DIR="$(cd "${THIS_SCRIPT_DIR}/../.." && pwd)"

echo "****************************************************************************************"
echo "Build and install MXnet..."
echo "****************************************************************************************"

if [[ -z "${NGRAPH_INSTALL_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${HOME}/ng_mx/ngraph_dist/lib/"
fi

cd "${MX_DIR}"
git submodule update --init
make USE_NGRAPH=1 NGRAPH_DIR=${NGRAPH_INSTALL_DIR} -j $(nproc)

if [ ! -f "./lib/libmxnet.so" ] ; then
  ( >&2 echo "FATAL ERROR: Can not found libmxnet.so. Exiting ...." )
  exit 1
else
   echo "Success to install ngraph-mxnet."
fi