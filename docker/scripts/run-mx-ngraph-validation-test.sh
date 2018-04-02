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
# Author:  Lam Nguyen

#!  /bin/bash

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

# This script is used to verify 2 models in image classicfications. 


if [ ! -z "${MX_NG_MODEL_DATASET}" ] ; then
    model_dataset="${MX_NG_MODEL_DATASET}"
fi
if [ ! -z "${1}" ] ; then
    model_dataset="${1}"
fi
if [ -z "${model_dataset}" ] ; then
    ( >&2 echo "SYNTAX ERROR: First and only parameter should be model-dataset." )
    ( >&2 echo "Supported model-dataset combinations are:")
    ( >&2 echo "    mlp-mnist  resnet110-cifar10")
    exit 1
fi

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately

# ===== run_MLP_MNIST() ========== 
# Function to run the example/image-classification/train_mnist.py
# Note:  read_data() will automatic download data from http://yann.lecun.com/exdb/mnist/ (train-images-idx3-ubyte.gz, t10k-images-idx3-ubyte.gz)
run_MLP_MNIST() {

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In train_mnist.py script, OMP_NUM_THREADS (omp_max_thread) and KMP_AFFINITY are explicitly
    # set only for the nGraph run by default. NUM_EPOCHS = 20 
    # Test parameters
    export TEST_MLP_MNIST_DATA_DIR="${dataDir}"
    export TEST_MLP_MNIST_LOG_DIR="${HOME}/ng-mx"
    # Run the test
    python -s example/image-classification/train_mnist.py --junit-xml=example/validation_tests_mlp_mnist_cpu.xml --junit-prefix=daily_validation_mlp_mnist_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_MLP_MNIST()

# ===== run_RESNET110_CIFAR10() ========== 
# Function to run the example/image-classification/train_cifar10.py
# Note: download_cifar10() will automatic download data from http://data.mxnet.io/data/cifar10

run_RESNET110_CIFAR10() {

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In train_cifar10.py script, OMP_NUM_THREADS (omp_max_thread) and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET110_CIFAR10_DATA_DIR="${dataDir}"
    export TEST_RESNET110_CIFAR10_LOG_DIR="${HOME}/ng-tx"
    export TEST_RESNET110_CIFAR10_EPOCHS="${MX_NG_EPOCHS:-}"
    if [ -z "${TEST_RESNET110_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET110_CIFAR10_EPOCHS=300  # Default is 300 epoches
    fi
    # Run the test
    python -s example/image-classification/train_cifar10.py --junit-xml=example/validation_tests_resnet110_cifar10_cpu.xml --junit-prefix=daily_validation_resnet110_cifar10_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_RESNET110_CIFAR10()


# ===== Main ==================================================================

export PYTHON_VERSION_NUMBER=2
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

export LD_LIBRARY_PATH="$HOME/ng-mx/ngraph_dist/lib"

echo "In $(basename ${0}):"
echo "  model_dataset=${model_dataset}"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ----- Install ngraph_dist ----------------------------------------------------

# Copy ngraph_dist into home directory, since right now ngraph-tensorflow-1.3
# is hard-coded to look for it in the home directory
if [ -d "$HOME/ngraph_dist" ] ; then
    ( >&2 echo "Directory $HOME/ngraph_dist already exists.  Removing it before installing the new version." )
    rm -fr "$HOME/ngraph_dist"
fi
echo "Copying ngraph_dist to $HOME/ngraph_dist"
cp -r ngraph_dist "$HOME/ngraph_dist"

# ----- Install Ngraph_Mxnet ---------------------------------------------------
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

if [ ! -f "./lib/libmxnet.so" ] ; then
  ( >&2 echo "FATAL ERROR: Can not found libmxnet.so. Exiting ...." )
  exit 1
fi

# ----- Run Models ----------------------------------
cd docker

case "${model_dataset}" in
mlp-mnist)  # Multi-Layer Perceptron (MLP) with MNIST dataset
    run_MLP_MNIST
    ;;
resnet110-cifar10)  # Resnet110 with CIFAR10 dataset
    run_RESNET110_CIFAR10
    ;;
*)
    ( >&2 echo "FATAL ERROR: ${model_dataset} is not supported in this script")
    exit 1
    ;;
esac

xtime="$(date)"
echo ' '
echo "===== Completed NGraph-MXNet Validation Test for ${model_dataset} at ${xtime} ====="
echo ' '

exit 0