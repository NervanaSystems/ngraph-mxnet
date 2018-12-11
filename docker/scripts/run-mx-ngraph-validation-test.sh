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

if [ ! -z "${MX_NG_MODEL}" ] ; then
    ng_mx_model="${MX_NG_MODEL}"
fi
if [ ! -z "${1}" ] ; then
    ng_mx_model="${1}"
fi
if [ -z "${ng_mx_model}" ] ; then
    ( >&2 echo "SYNTAX ERROR: First and only parameter should be ng_mx_model." )
    ( >&2 echo "Supported ng_mx_model ( Ngraph MXNET model/network) combinations are:")
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
    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '
    virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
    source "${venv_dir}/bin/activate"
    cd python && pip install -e . && pip install psutil && pip install pytest && cd ../
    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In train_mnist.py script, OMP_NUM_THREADS (omp_max_thread) and KMP_AFFINITY are explicitly
    # set only for the nGraph run by default. NUM_EPOCHS = 20 
    # Test parameters
    export TEST_MLP_MNIST_LOG_DIR="${HOME}/ng-mx"
    # Run the test
    pytest -s docker/scripts/test_mnist_cpu_daily_validation.py --junit-xml=validation_tests_mnist_mlp_cpu.xml --junit-prefix=daily_validation_mnist_mlp_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_MLP_MNIST()

# ===== run_RESNET110_CIFAR10() NEED TO DO ========== 
# Function to run the example/image-classification/train_cifar10.py
# Note: download_cifar10() will automatic download data from http://data.mxnet.io/data/cifar10

run_RESNET110_CIFAR10() {
    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '
    virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
    source "${venv_dir}/bin/activate"
    cd python && pip install -e . && pip install psutil && pip install pytest && cd ../
    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In train_cifar10.py script, OMP_NUM_THREADS (omp_max_thread) and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET_CIFAR10_LOG_DIR="${HOME}/ng-mx"
    export TEST_MX_NG_RESNET_NUM_LAYERS="${MX_NG_RESNET_NUM_LAYERS}"
    export TEST_MX_RESNET_NUM_CLASSES="${MX_NG_RESNET_NUM_CLASSES}"
    export TEST_MX_NG_RESNET_NUM_EXAMPLES="${MX_NG_RESNET_NUM_EXAMPLES}"
    export TEST_MX_NG_RESNET_IMAGE_SHAPE="${MX_NG_RESNET_IMAGE_SHAPE}"
    export TEST_MX_NG_RESNET_PAD_SIZE="${MX_NG_RESNET_PAD_SIZE}"
    export TEST_MX_NG_RESNET_BATCH_SIZE="${MX_NG_RESNET_BATCH_SIZE}"
    export TEST_MX_NG_RESNET_LR="${MX_NG_RESNET_LR}"
    export TEST_MX_NG_RESNET_LR_STEP_EPOCHS="${MX_NG_RESNET_LR_STEP_EPOCHS}"
    export TEST_MX_NG_RESNET_WITH_NNP="${MX_NG_RESNET_WITH_NNP}"
    export TEST_RESNET110_CIFAR10_EPOCHS="${MX_NG_EPOCHS:-}"
    export TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY="${MX_NG_RESNET_ACCEPTABLE_ACCURACY}"
    export TEST_MAKE_VARIABLES="${MAKE_VARIABLES}"
    if [ -z "${TEST_RESNET110_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET110_CIFAR10_EPOCHS=1  # Default is 300 epoches
    fi
    # Run the test
    pytest -s docker/scripts/test_resnet_cifar_daily_validation.py --junit-xml=validation_tests_resnet_cifar_cpu.xml --junit-prefix=daily_validation_resnet_cifar_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_RESNET110_CIFAR10()

run_RESNET_I1K() {
    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '
    virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
    source "${venv_dir}/bin/activate"
    cd python && pip install -e . && pip install psutil && pip install pytest && cd ../
    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In train_cifar10.py script, OMP_NUM_THREADS (omp_max_thread) and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET_I1K_LOG_DIR="${HOME}/ng-mx"
    export TEST_MX_NG_RESNET_NUM_LAYERS="${MX_NG_RESNET_NUM_LAYERS}"
    export TEST_MX_RESNET_NUM_CLASSES="${MX_NG_RESNET_NUM_CLASSES}"
    export TEST_MX_NG_RESNET_NUM_EXAMPLES="${MX_NG_RESNET_NUM_EXAMPLES}"
    export TEST_MX_NG_RESNET_IMAGE_SHAPE="${MX_NG_RESNET_IMAGE_SHAPE}"
    export TEST_MX_NG_RESNET_PAD_SIZE="${MX_NG_RESNET_PAD_SIZE}"
    export TEST_MX_NG_RESNET_BATCH_SIZE="${MX_NG_RESNET_BATCH_SIZE}"
    export TEST_MX_NG_RESNET_DATA_DIR="${MX_NG_RESNET_DATA_DIR}"
    export TEST_MX_NG_RESNET_LR="${MX_NG_RESNET_LR}"
    export TEST_MX_NG_RESNET_LR_STEP_EPOCHS="${MX_NG_RESNET_LR_STEP_EPOCHS}"
    export TEST_MX_NG_RESNET_WITH_NNP="${MX_NG_RESNET_WITH_NNP}"
    export TEST_RESNET_I1K_EPOCHS="${MX_NG_EPOCHS:-}"
    export TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY="${MX_NG_RESNET_ACCEPTABLE_ACCURACY}"
    export TEST_MAKE_VARIABLES="${MAKE_VARIABLES}"
    if [ -z "${TEST_RESNET_I1K_EPOCHS}" ] ; then
        export TEST_RESNET_I1K_EPOCHS=1  # Default is 300 epoches
    fi
    # Run the test
    pytest -s docker/scripts/test_resnet_i1k_daily_validation.py --junit-xml=validation_tests_resnet_i1k_cpu.xml --junit-prefix=daily_validation_resnet_i1k_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_RESNET_I1K()

# ===== Main ==================================================================
echo "the Python version in run_mx_ngraph-validation.py is: PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

echo "In $(basename ${0}):"
echo "  ng_mx_model=${ng_mx_model}"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"

# ----- Install ngraph_dist ----------------------------------------------------
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
export MAKE_VARIABLES="${MAKE_VARIABLES}"
export NGRAPH_BRANCH="${NGRAPH_BRANCH}"
export OS_SYSTEM="${OS_SYSTEM}"
./build-install-mx.sh 2>&1 | tee ../mx-build.log
echo "===== Build & Install Pipeline Exited with $? and endtime ${xtime} ===="

# ----- Run Models ----------------------------------
cd "$HOME/ng-mx/"

case "${ng_mx_model}" in
mlp-mnist)  # Multi-Layer Perceptron (MLP) with MNIST dataset
    run_MLP_MNIST
    ;;
resnet110-cifar10)  # Resnet110 with CIFAR10 dataset
    run_RESNET110_CIFAR10
    ;;
resnet-i1k)  # Resnet with I1K dataset
    run_RESNET_I1K
    ;;
*)
    ( >&2 echo "FATAL ERROR: ${ng_mx_model} is not supported in this script")
    exit 1
    ;;
esac

xtime="$(date)"
echo ' '
echo "===== Completed NGraph-MXNet Validation Test for ${ng_mx_model} at ${xtime} ====="
echo ' '

exit 0
