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
export PYTHON_VERSION_NUMBER=3

export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# We don't need ccache because we are building in a (fresh) docker container
export BUILD_MX_WITH_CCACHE=0

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-mx-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"

xtime="$(date)"
echo  ' '
echo  "===== Running unit test  at ${xtime} ====="
echo  ' '
cd "$HOME/ng-mx/"
PS1='prompt> '
PS2='prompt-more> '
virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
source "${venv_dir}/bin/activate"
cd python  && pip install pylint cpplint && cd ../
make cpplint |& tee  check_style.txt
STYLE_CHECK_LOGFILE='check_style.txt'
if [ "$(grep 'All passed!' ${STYLE_CHECK_LOGFILE} | wc -l)" = "1" ] ; then
	echo "Pass to check style. Continue running the unit tests"
else
	echo "Fail to check the style. Exiting ..."
	exit 1
fi
cd "$HOME/ng-mx/docker/scripts/"
./run-unit-tests.sh 2>&1 | tee ../mx-tests.log
echo "===== Unit Tests Pipeline Exited with $? ====="

xtime="$(date)"
echo ' '
echo "===== Completed MXnet-NGraph-Unittes Build and Test at ${xtime} ====="
echo ' '
