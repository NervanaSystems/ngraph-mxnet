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

# This script is used to verify benchmark for ngraph_mxnet

echo "Build Mxnet_Ngraph"

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediatel


# ===== run_INCEPTION_V4() ========== 
# Function to run the  OMP_NUM_THREADS=1 KMP_AFFINITY=granularity=fine,compact,1,0 ./bmark.sh --network inception-v4 --batch-size 128
run_INCEPTION_V4() {
    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '
    virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
    source "${venv_dir}/bin/activate"
    cd python && pip install -e . && pip install psutil && pip install pytest && cd ../
    xtime="$(date)"
    echo  ' '
    echo  "===== Running Ngraph Mxnet DeepMark on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # Test parameters
    export TEST_DEEPMARK_LOG_DIR="${HOME}/ng-mx/mxnet-deepmark/image+video/"
    export TEST_OMP_NUM_THREADS="${MX_OMP_NUM_THREADS}"
    export TEST_KMP_BLOCKTIME="${MX_NG_KMP_BLOCKTIME}"
    export TEST_BATCH_SIZE="${MX_NG_BATCH_SIZE}"
    export TEST_KMP_AFFINITY="${MX_NG_KMP_AFFINITY}"

    # Run the inception_v4
    #cmd="pytest -s docker/scripts/test_deepmark_inception_v4_inference.py --junit-xml=validation_test_deepmark_inception_v4_inference.xml --junit-prefix=inference_deepmark_inception_v4_cpu"
    #eval $cmd

    # Run the resnet_50
    #cmd="pytest -s docker/scripts/test_deepmark_resnet_50_inference.py --junit-xml=validation_test_deepmark_resnet_50_inference.xml --junit-prefix=inference_deepmark_resnet_50_cpu"
    #eval $cmd

    # Run the a3c
    #cmd="pytest -s docker/scripts/test_deepmark_a3c_inference.py --junit-xml=validation_test_deepmark_a3c_inference.xml --junit-prefix=inference_deepmark_a3c_cpu"
    #eval $cmd

    # Run the test wide_deep
    #cmd="pytest -s docker/scripts/test_deepmark_wide_deep_inference.py --junit-xml=validation_test_deepmark_wide_deep_inference.xml --junit-prefix=inference_deepmark_wide_deep_cpu"
    #eval $cmd

    # Run the test mobilenet
    #cmd="pytest -s docker/scripts/test_deepmark_mobilenet_inference.py --junit-xml=validation_test_deepmark_mobilenet_inference.xml --junit-prefix=inference_deepmark_mobilenet_cpu"
    #eval $cmd

    # Run the densenet121
    cmd="pytest -s docker/scripts/test_deepmark_densenet121_inference.py --junit-xml=validation_test_deepmark_densenet121_inference.xml --junit-prefix=inference_deepmark_densenet121_cpu"
    eval $cmd

    # Run the Faster-RCNN, --batch-size 1
    #cmd="pytest -s docker/scripts/test_deepmark_Faster_RCNN_inference.py --junit-xml=validation_test_deepmark_Faster_RCNN_inference.xml --junit-prefix=inference_deepmark_Faster_RCNN_cpu"
    #eval $cmd

    # Run the squeezenet1.1
    cmd="pytest -s docker/scripts/test_deepmark_squeezenet_cpu_inference.py --junit-xml=validation_test_deepmark_squeezenet_inference.xml --junit-prefix=inference_deepmark_squeezenet_cpu"
    eval $cmd

    # Run DCGAN

    # Run Transformer 



    echo "===== Inference CPU-Backend Pipeline Exited with $? ====="

}  # run_INCEPTION_V4()


# ===== Main ==================================================================

echo "the Python version in run_mx_ngraph-validation.py is: PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

export LD_LIBRARY_PATH="$HOME/ng-mx/ngraph_dist/lib"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ----- Run Models ----------------------------------
cd "$HOME/ng-mx/"

echo "Run run_INCEPTION_V4()"

run_INCEPTION_V4


xtime="$(date)"
echo ' '
echo "===== Completed NGraph-MXNet Inference at ${xtime} ====="
echo ' '

exit 0