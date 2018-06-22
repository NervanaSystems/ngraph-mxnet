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

echo "Run deepmark"

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately

# ===== Main ==================================================================

echo "the Python version in run_mx_ngraph-validation.py is: PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

export LD_LIBRARY_PATH="$HOME/ng-mx/ngraph_dist/lib"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ----- Install ngraph_dist ----------------------------------------------------

# Copy ngraph_dist into home directory, since right now ngraph-tensorflow-1.3
# is hard-coded to look for it in the home directory
if [ -d "$HOME/ngraph_dist" ] ; then
    ( >&2 echo "Directory $HOME/ngraph_dist already exists.  Removing it before installing the new version." )
    rm -fr "$HOME/ngraph_dist"
fi
echo "Copying ngraph_dist to $HOME/ngraph_dist"
cp -r ngraph_dist "$HOME/ngraph_dist"

# ----- Install Ngraph_Mxnet ---------------------------------------------------
cd "$HOME/ng-mx/docker/scripts/"
xtime="$(date)"
echo  ' '
echo  "===== Configuring Mxnet Build at ${xtime} ====="
echo  ' '
./config-mx.sh 2>&1 | tee ../mx-config.log
echo  "===== Configuring Mxnet Build Exited with $? ====="

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

# ----- Sanity Checks ----------------------------------------------------------

if [ ! -f "$LD_LIBRARY_PATH/libngraph.so" ] ; then
  ( >&2 echo "FATAL ERROR: libngraph.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
  exit 1
fi

if [ ! -f "$LD_LIBRARY_PATH/libmkldnn.so" ] ; then
  ( >&2 echo "FATAL ERROR: libmkldnn.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
  exit 1
fi