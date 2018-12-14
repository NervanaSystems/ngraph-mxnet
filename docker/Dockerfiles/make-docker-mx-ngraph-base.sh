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

if [ "${OS_SYSTEM}" = "CENTOS7" ] ; then
  DOCKER_FILE='Dockerfile.ci.centos7.mxnet'
  IMAGE_NAME='ngmx_ci_centos7'
elif [ "${OS_SYSTEM}" = "UBUNTU16.4" ] ; then
  DOCKER_FILE='Dockerfile.ci.mxnet'
  IMAGE_NAME='ngmx_ci_ubuntu16_4'
else
  echo "Need to provide the OS_SYSTEM. Exitting ..."
  exit 1
fi

# The docker image ID will passed
IMAGE_ID="${1}"
#IMAGE_ID="$(git rev-parse HEAD)"

if [ -z "${IMAGE_ID}" ] ; then
    echo 'Missing an image version as the only argument. Exitting ...'
    exit 1
else
    echo "IMAGE_ID= ${IMAGE_ID}"
fi

if [ -z "${GCC_VERSION}" ] ; then
    GCC_VERSION="gcc-5"
fi

# If there are more parameters, which are intended to be directly passed to
# the "docker build ..." command-line, then shift off the IMAGE_NAME
if [ "x${2}" = 'x' ] ; then
    shift
fi

set -u  # No unset variables

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

# Context is the docker directory, to avoid including all of
# ngraph-mxnet in the context.

# The $@ allows us to pass command-line options easily to docker build.
# Note that a "shift" is done above to remove the IMAGE_ID from the cmd line.

docker build  --rm=true \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} ${GCC_VERSION} \
       $@ \
       -f="${DOCKER_FILE}"  -t="${IMAGE_NAME}:${IMAGE_ID}"   ..