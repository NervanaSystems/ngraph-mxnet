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

if [ -z "${1}" ] ; then
    export PYTHON_VERSION_NUMBER="2"  # Build for Python 3 by default
else
    export PYTHON_VERSION_NUMBER="${1}"
fi
echo "======PYTHON_VERSION_NUMBER========"
echo "${PYTHON_VERSION_NUMBER}"

# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).

#IMAGE_ID="$(git rev-parse HEAD)"

#if [ -z "${IMAGE_ID}" ] ; then
#    echo 'Missing an image version as the only argument. Exitting ...'
#    exit 1
#fi

IMAGE_NAME='ngmx_ci'
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Missing an image version as the only argument. Exitting ...'
    exit 1
fi

set -u  # No unset variables after this point

ngraph_mx_dir="$(realpath ../..)"

docker_mx_dir="/home/dockuser/ng-mx"

#script='run-mx-ngraph-benchmark-test.sh'
script='run-ng-mx-deepmark-test.sh'

# Parameters:
#           MX_NG_RUN_PER_SCRIPT          Model to run


## benchmark.py
#--networks NETWORKS [NETWORKS ...]
#                        one or more networks in the format
#                        mode:network_name:batch_size:image_size The
#                        network_name is a valid model defined as
#                        network_name.py in the image-classification/symbol
#                        folder for native imagenet Or a gluon vision model
#                        defined in
#                        mxnet/python/mxnet/gluon/model_zoo/model_store.py.
# --worker_file WORKER_FILE
#                        file that contains a list of worker hostnames or list
#                        of worker ip addresses that can be sshed without a
#                        password.
#  --worker_count WORKER_COUNT
#                        number of workers to run benchmark on.
#  --gpu_count GPU_COUNT
#                        number of gpus on each worker to use.

## benchmark_score.py
docker run --rm \
      --env RUN_UID="$(id -u)" \
      --env RUN_CMD="${docker_mx_dir}/docker/scripts/${script}" \
      --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}"\
      --env http_proxy=http://proxy-fm.intel.com:911 \
      --env https_proxy=http://proxy-fm.intel.com:912 \
      -v "${ngraph_mx_dir}:${docker_mx_dir}" \
      "${IMAGE_NAME}:${IMAGE_ID}" /home/run-as-user.sh
