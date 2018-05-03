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

#!  /bin/bash
#Author:  Lam Nguyen

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables

# For now we simply build ng-mx for python 2.  Later, python 3 builds will
# be added.
export PYTHON_VERSION_NUMBER=2
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# We don't need ccache because we are building in a (fresh) docker container
export BUILD_MX_WITH_CCACHE=0

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-mx-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

# ngraph_dist (with libmkldnn.so) is expected to be at top of ngraph-mxnet
export LD_LIBRARY_PATH="$HOME/ng-mx/ngraph_dist/lib"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Copy ngraph_dist into home directory
if [ -d "$HOME/ngraph_dist" ] ; then
    ( >&2 echo "Directory $HOME/ng-mx/ngraph_dist already exists.  Removing it before installing the new version." )
    rm -fr "$HOME/ngraph_dist"
fi

echo "Copying ngraph_dist to $HOME/ngraph_dist"
cp -r ngraph_dist "$HOME/ngraph_dist"

if [ ! -f "$LD_LIBRARY_PATH/libngraph.so" ] ; then
  ( >&2 echo "FATAL ERROR: libngraph.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
  exit 1
fi

if [ ! -f "$LD_LIBRARY_PATH/libmkldnn.so" ] ; then
  ( >&2 echo "FATAL ERROR: libmkldnn.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
  exit 1
fi

# Make sure the Bazel cache is in /tmp, as docker images have too little space
# in the root filesystem, where /home (and $HOME/.cache) is.  Even though we
# may not be using the Bazel cache in the builds (in docker), we do this anyway
# in case we decide to turn the Bazel cache back on.
echo "Adjusting bazel cache to be located in /tmp/bazel-cache"
rm -fr "$HOME/.cache"
mkdir /tmp/bazel-cache
ln -s /tmp/bazel-cache "$HOME/.cache"

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

# --------------------------------------------------------------------------------

xtime="$(date)"
echo  ' '
echo  "===== Running unit test  at ${xtime} ====="
echo  ' '
cd "$HOME/ng-mx/"
PS1='prompt> '
PS2='prompt-more> '
virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
source "${venv_dir}/bin/activate"
cd "$HOME/ng-mx/docker/scripts/"
export PYTHONPATH=$PYTHONPATH:/tmp/venv_python${PYTHON_VERSION_NUMBER}/bin/python${PYTHON_VERSION_NUMBER}
eho '${PYTHONPATH}'
./run-ng-mx-unit-tests.sh 2>&1 | tee ../mx-tests.log
echo "===== Unit Tests Pipeline Exited with $? ====="

xtime="$(date)"
echo ' '
echo "===== Completed MXnet-NGraph-Unittes Build and Test at ${xtime} ====="
echo ' '
