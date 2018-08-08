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
# Author:  Lam Nguyen

#!  /bin/bash

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

# This script is used to verify benchmark for ngraph_mxnet

echo "Build Mxnet_Ngraph"

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediatel

# ===== Main ==================================================================

echo "the Python version in run_mx_ngraph-validation.py is: PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"

cd "$HOME/ng-mx/docker/scripts/"

xtime="$(date)"
echo  ' '
echo  "===== Building and Installing Mxnet at ${xtime} ====="
echo  ' '
# Make sure pip install uses sudo, for installing into system
# In addition, pip seems to ignore http_proxy env vars, so
# explicitly set them here
export PIP_INSTALL_FROM_SUDO=1
export PIP_INSTALL_EXTRA_ARGS="--proxy=$http_proxy --proxy=$https_proxy"
./build-install-mx.sh 2>&1 | tee ../mx-build.log
echo "===== Build & Install Pipeline Exited with $? and endtime ${xtime} ===="



