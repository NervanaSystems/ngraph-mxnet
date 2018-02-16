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
set -u  # No unset variables

DOCKER_FILE='Dockerfile.ci.mxnet'

# The docker image ID is currently just the git SHA of this cloned repo
IMAGE_ID="$(git rev-parse HEAD)"

# Context is the docker directory, to avoid including all of
# ngraph-mxnet in the context.
docker build  --rm=true  --build-arg http_proxy=http://proxy-us.intel.com:911  --build-arg https_proxy=https://proxy-us.intel.com:911  -f="${DOCKER_FILE}"  -t="mxnet:${IMAGE_ID}"   ..