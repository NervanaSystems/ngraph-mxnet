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

set -e  # Fail on any command with non-zero exit

ngraph_mx_dir="$(realpath ../..)"

# The docker image ID is currently just the git SHA of this cloned repo.
# We need this ID to know which docker image to run with.
# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).
IMAGE_ID="$(git rev-parse HEAD)"

docker_mx_dir="/home/dockuser/ng-mx"

script='run-mx-ngraph-validation-test.sh'

# The docker image ID is currently just the git SHA of this cloned repo.
# We need this ID to know which docker image to run with.
# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).
IMAGE_ID="$(git rev-parse HEAD)"

# Parameters:
#           MX_NG_MODEL          Model to run
#           MX_NG_ITERATIONS     Number of iterations (aka steps) to run
#           MX_NG_DO_NOT_RUN     If defined and not empty, does not run pytest

# MX_NG_MODEL *must* be defined to run any validation test in docker
if [ -z "${MX_NG_MODEL}" ] ; then
    ( >&2 echo "MX_NG_MODEL must be set to the model to run" )
    exit
fi

docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${docker_mx_dir}/docker/scripts/${script}" \
       --env MX_NG_MODEL="${MX_NG_MODEL}" \
       --env MX_NG_ITERATIONS="${MX_NG_ITERATIONS}" \
       --env MX_NG_EPOCHS="${MX_NG_EPOCHS}" \
       --env MX_NG_DO_NOT_RUN="${MX_NG_DO_NOT_RUN}" \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       -v "${ngraph_mx_dir}:${docker_mx_dir}" \
       "mx_ngraph_base:${IMAGE_ID}" /home/run-as-user.sh
