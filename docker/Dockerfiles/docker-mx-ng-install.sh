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

# Script parameters:
#
# $1 IMAGE_ID    Required: ID of the ngmx_ci docker image to use
# $2 PythonVer  Optional: version of Python to build with (default: 2)

set -e  # Fail on any command with non-zero exit

# Get the python version

if [ -z "${2}" ] ; then
    export PYTHON_VERSION_NUMBER="3"  # Build for Python 3 by default
else
    export PYTHON_VERSION_NUMBER="${2}"
fi
echo "======PYTHON_VERSION_NUMBER========"
echo " PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"

echo "======MAKE_VARIABLES========"
echo " MAKE_VARIABLES= ${MAKE_VARIABLES}"

# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).

if [[ ${MAKE_VARIABLES} == "USE_CUDA" ]]; then
    IMAGE_NAME='ngmx_ci_gpu'
else
    IMAGE_NAME='ngmx_ci'
fi

IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Missing an image version as the only argument. Exitting ...'
    exit 1
fi

set -u  # No unset variables after this point

ngraph_mx_dir="$(realpath ../..)"

docker_mx_dir="/home/dockuser/ng-mx"

script='run-ng-mx-install.sh'

docker run --no-cache --rm \
      --env RUN_UID="$(id -u)" \
      --env RUN_CMD="${docker_mx_dir}/docker/scripts/${script}" \
      --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}"\
      --env MAKE_VARIABLES="${MAKE_VARIABLES}" \
      --env http_proxy=http://proxy-fm.intel.com:911 \
      --env https_proxy=http://proxy-fm.intel.com:912 \
      -v "${ngraph_mx_dir}:${docker_mx_dir}" \
      "${IMAGE_NAME}:${IMAGE_ID}" /home/run-as-user.sh
