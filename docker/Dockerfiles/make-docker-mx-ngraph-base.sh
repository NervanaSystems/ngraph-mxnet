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

#!  /bin/sh
# Author:  Lam Nguyen

# This script is designed to be run from the docker/Dockerfiles directory

set -e  # Fail on any command with non-zero exit

DOCKER_FILE='Dockerfile.ci.mxnet'

# The docker image name
#IMAGE_NAME="ngmx_ci"

# The docker image ID will passed
#IMAGE_ID="${1}"
IMAGE_ID="$(git rev-parse HEAD)"
echo "IMAGE_ID= ${IMAGE_ID}"

#if [ -z "${IMAGE_ID}" ] ; then
#    echo 'Missing an image version as the only argument. Exitting ...'
#    exit 1
#fi

set -u  # No unset variables

# Context is the docker directory, to avoid including all of
# ngraph-mxnet in the context.
#docker build  --rm=true  --build-arg http_proxy=http://proxy-us.intel.com:911  --build-arg https_proxy=https://proxy-us.intel.com:911  -f="${DOCKER_FILE}"  -t="${IMAGE_NAME}:${IMAGE_ID}"   ..
docker build --rm=true  --build-arg http_proxy=http://proxy-fm.intel.com:911  --build-arg https_proxy=http://proxy-fm.intel.com:912  -f="${DOCKER_FILE}"  -t="mxnet:${IMAGE_ID}"   ..
