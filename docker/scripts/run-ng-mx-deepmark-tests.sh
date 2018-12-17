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


# ===== run_inference_topologies() ========== 
# Function to run the  OMP_NUM_THREADS=1 KMP_AFFINITY=granularity=fine,compact,1,0 ./bmark.sh --network inception-v4 --batch-size 128
run_inference_topologies() {
    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '
    virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
    source "${venv_dir}/bin/activate"
    cd python && pip install -e . && pip install psutil pytest scipy gluoncv && cd ../
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
    export TEST_DEEPMARK_TYPE="${MX_NG_DEEPMARK_TYPE}"
    export LD_LIBRARY_PATH="${HOME}/ng-mx/warp-ctc/build"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    
    INFERENCE_PY_SCRIPTS="${HOME}/jenkins/ngraph-mxnet-validation/ng-mx-topologies-scripts/"

    echo "====DEBUG ======="
    echo " INFERENCE_PY_SCRIPTS ===== ${INFERENCE_PY_SCRIPTS}"
    echo `ls ${INFERENCE_PY_SCRIPTS}`
    echo "=== END DEBUG ===="

    if [ "${TEST_BATCH_SIZE}" == "1" ] ; then
        # 23. Run the Faster-RCNN, --batch-size 1
        cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_Faster_RCNN_inference.py --junit-xml=validation_test_deepmark_Faster_RCNN_inference.xml --junit-prefix=inference_deepmark_Faster_RCNN_cpu"
        eval $cmd
    else
        echo "Faster-RCNN doesn't work with any --batch-size except 1."
    fi

    echo "===== Inference CPU-Backend Pipeline Exited with $? ====="

}  # run_inference_topologies()


# ===== Main ==================================================================
if [ "${OS_SYSTEM}" = "CENTOS7" ] ; then
    export PYTHON_VERSION_NUMBER=3.6
else
    export PYTHON_VERSION_NUMBER=3
fi

echo "the Python version in run_mx_ngraph-validation.py is: PYTHON_VERSION_NUMBER = ${PYTHON_VERSION_NUMBER}"
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"
export venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-tf-ng-build-as-user.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/ng-mx"

echo "In $(basename ${0}):"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"

# ----- Run Models ----------------------------------
cd "$HOME/ng-mx/"

echo "Run run_inference_topologies()"

run_inference_topologies

xtime="$(date)"
echo ' '
echo "===== Completed NGraph-MXNet Inference at ${xtime} ====="
echo ' '

exit 0
