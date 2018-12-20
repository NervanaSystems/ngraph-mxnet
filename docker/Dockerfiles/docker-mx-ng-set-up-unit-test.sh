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
# Author:  Lam Nguyen

# This script is designed to be run from the docker directory

set -e  # Fail on any command with non-zero exit
# Get the python version

if [ -z "${2}" ] ; then
    export PYTHON_VERSION_NUMBER="3"  # Build for Python 3 by default
else
    export PYTHON_VERSION_NUMBER="${2}"
fi
echo "======PYTHON_VERSION_NUMBER========"
echo " PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"

# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).
D_CMD="docker"
if [[ ${MAKE_VARIABLES} == "USE_CUDA" ]]; then
    IMAGE_NAME='ngmx_ci_gpu'
    D_CMD="nvidia-docker"
fi

if [ "${OS_SYSTEM}" = "CENTOS7" ]; then
    IMAGE_NAME='ngmx_ci_centos7'
elif [ "${OS_SYSTEM}" = "UBUNTU16.4" ]; then
    IMAGE_NAME='ngmx_ci_ubuntu16_4'
else
    echo "Missing Input Parameters : MAKE_VARIABLES = ${MAKE_VARIABLES}, and OS_SYSTEM = ${OS_SYSTEM}. Existing .."
    exit 1
fi

IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Missing an image version as the only argument. Exitting ...'
    exit 1
fi
set -u  # No unset variables

ngraph_mx_dir="$(realpath ../..)"

# The docker image ID is currently just the git SHA of this cloned repo.
# We need this ID to know which docker image to run with.
# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).

${D_CMD} run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD='/home/dockuser/ng-mx/docker/scripts/set-up-unit-test.sh' \
       --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}" \
       --env MAKE_VARIABLES="${MAKE_VARIABLES}" \
       --env http_proxy=http://proxy-fm.intel.com:911 \
       --env https_proxy=http://proxy-fm.intel.com:912 \
       -v "${ngraph_mx_dir}:/home/dockuser/ng-mx" \
       "${IMAGE_NAME}:${IMAGE_ID}" /home/run-as-user.sh
