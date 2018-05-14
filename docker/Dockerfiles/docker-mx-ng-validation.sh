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
    export PYTHON_VERSION_NUMBER="2"  # Build for Python 3 by default
else
    export PYTHON_VERSION_NUMBER="${2}"
fi

# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).
#IMAGE_NAME='ngmx_ci'
#IMAGE_ID="${1}"

IMAGE_ID="$(git rev-parse HEAD)"

if [ -z "${IMAGE_ID}" ] ; then
    echo 'Missing an image version as the only argument. Exitting ...'
    exit 1
fi

ngraph_mx_dir="$(realpath ../..)"

docker_mx_dir="/home/dockuser/ng-mx"

script='run-mx-ngraph-validation-test.sh'

# Parameters:
#           MX_NG_MODEL          Model to run
#           MX_NG_ITERATIONS     Number of iterations (aka steps) to run
#           MX_NG_DO_NOT_RUN     If defined and not empty, does not run pytest

# MX_NG_MODEL *must* be defined to run any validation test in docker
if [ -z "${MX_NG_MODEL}" ] ; then
    ( >&2 echo "MX_NG_MODEL must be set to the model to run" )
    exit
fi

## RESNET : Parameters and its default values
#network = 'resnet'
#num_layers = 110
#num_classes = 10
#num_examples = 50000
#image_shape = '3,28,28'
#pad_size = 4
#batch_size = 128
#num_epochs = 1
#lr = .05
#lr_step_epochs = '200,250'
#--with-nnp : False

## MLP_MNIST : Parameters and its default values.
#gpus = None
#batch_size = 64
#disp_batches = 100
#num_epochs = 20
#lr = .05
#lr_step_epochs = 10
#--num-examples type=int, default=60000
#--num-classes type=int default=10
#--with-nnp : False
#--add_stn : False

## NEED TO DO. 

docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${docker_mx_dir}/docker/scripts/${script}" \
       --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}" \
       --env MX_NG_MODEL="${MX_NG_MODEL}" \
       --env MX_NG_ITERATIONS="${MX_NG_ITERATIONS}" \
       --env MX_NG_EPOCHS="${MX_NG_EPOCHS}" \
       --env MX_NG_DO_NOT_RUN="${MX_NG_DO_NOT_RUN}" \
       --env MX_NG_RESNET_NUM_LAYERS="${MX_NG_RESNET_NUM_LAYERS}" \
       --env MX_NG_RESNET_NUM_CLASSES="${MX_NG_RESNET_NUM_CLASSES}" \
       --env MX_NG_RESNET_NUM_EXAMPLES="${MX_NG_RESNET_NUM_EXAMPLES}" \
       --env MX_NG_RESNET_IMAGE_SHAPE="${MX_NG_RESNET_NUM_IMAGE_SHAPE}" \
       --env MX_NG_RESNET_PAD_SIZE="${MX_NG_RESNET_PAD_SIZE}" \
       --env MX_NG_RESNET_BATCH_SIZE="${MX_NG_RESNET_BATCH_SIZE}" \
       --env MX_NG_RESNET_LR="${MX_NG_RESNET_LR}" \
       --env MX_NG_RESNET_LR_STEP_EPOCHS="${MX_NG_RESNET_LR_STEP_EPOCHS}" \
       --env MX_NG_RESNET_WITH_NNP="${MX_NG_RESNET_WITH_NNP}" \
       --env MX_NG_RESNET_ACCEPTABLE_ACCURACY="${MX_NG_RESNET_ACCEPTABLE_ACCURACY}" \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       -v "${ngraph_mx_dir}:${docker_mx_dir}" \
       "mxnet:${IMAGE_ID}" /home/run-as-user.sh
