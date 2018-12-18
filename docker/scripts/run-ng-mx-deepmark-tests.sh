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

    # 1. Run the Faster-RCNN, --batch-size 1
    if [ "${TEST_BATCH_SIZE}" == "1" ] ; then

        cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_Faster_RCNN_inference.py --junit-xml=validation_test_deepmark_Faster_RCNN_inference.xml --junit-prefix=inference_deepmark_Faster_RCNN_cpu"
        eval $cmd
    else
        echo "Faster-RCNN doesn't work with any --batch-size except 1."
    fi

    ## 2. DeepSpeech 2: Issue NGRAPH-2911
    if [ "${TEST_BATCH_SIZE}" == "1" ] ; then
        #24. Run DeepSpeed 2
        cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_deepspeech_inference.py --junit-xml=validation_test_deepmark_deepspeech_inference.xml --junit-prefix=inference_deepmark_deepspeech"
        eval $cmd
    else
        echo "DeepSpeech 2 doesn't work with any --batch-size except 1."
    fi

    ## 3. Mask_rcnn_tusimple
    if [ "${TEST_BATCH_SIZE}" == "1" ] ; then
        #Dowload Data
        cd "mxnet-deepmark/image+video/maskrcnn_tusimple/"
        cp -r /dataset/cityscape/maskrcnn-tusimple/data .
        pip install -r requirements.txt
        make

        cd "${HOME}/ng-mx/mxnet-deepmark"
        cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_mask_rcnn_tusimple_cpu_backend.py --junit-xml=validation_test_deepmark_mask_rcnn_tusimple_inference.xml --junit-prefix=inference_deepmark_mask_rcnn_tusimple_cpu"
        eval $cmd
    else
        echo "MaskRCNN tusimple ONLY supports --batch-size 1."
    fi

    # 4. Run the inception_v4
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_inception_v4_inference.py --junit-xml=validation_test_deepmark_inception_v4_inference.xml --junit-prefix=inference_deepmark_inception_v4_cpu"
    eval $cmd

    # 5. Run the inception_v3
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_inception_v3_inference.py --junit-xml=validation_test_deepmark_inception_v3_inference.xml --junit-prefix=inference_deepmark_inception_v3_cpu"
    eval $cmd

    # 6. Run the inception_resnet_v2
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_inception_resnet_v2_inference.py --junit-xml=validation_test_deepmark_inception_resnet_v2_inference.xml --junit-prefix=inference_deepmark_inception_resnet_v2_cpu"
    eval $cmd

    # 7. Run the resnet_50_v2
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_resnet_50_inference.py --junit-xml=validation_test_deepmark_resnet_50_v2_inference.xml --junit-prefix=inference_deepmark_resnet_50_v2_cpu"
    eval $cmd

    # 8. Run the resnet_50_v1
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_resnet_50_v1_inference.py --junit-xml=validation_test_deepmark_resnet_50_v1_inference.xml --junit-prefix=inference_deepmark_resnet_50_v1_cpu"
    eval $cmd

    # 9. Run the a3c
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_a3c_inference.py --junit-xml=validation_test_deepmark_a3c_inference.xml --junit-prefix=inference_deepmark_a3c_cpu"
    eval $cmd

    # 10. Run the test wide_deep
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_wide_deep_inference.py --junit-xml=validation_test_deepmark_wide_deep_inference.xml --junit-prefix=inference_deepmark_wide_deep_cpu"
    eval $cmd

    # 11. Run the test mobilenet
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_mobilenet_inference.py --junit-xml=validation_test_deepmark_mobilenet_inference.xml --junit-prefix=inference_deepmark_mobilenet_cpu"
    eval $cmd

    # 12. Run the test mobilenet_v2
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_mobilenet_v2_inference.py --junit-xml=validation_test_deepmark_mobilenet_v2_inference.xml --junit-prefix=inference_deepmark_mobilenet_v2_cpu"
    eval $cmd

    # 13. Run the densenet121
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_densenet121_inference.py --junit-xml=validation_test_deepmark_densenet121_inference.xml --junit-prefix=inference_deepmark_densenet121_cpu"
    eval $cmd

    # 14. Run the densenet161
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_densenet161_inference.py --junit-xml=validation_test_deepmark_densenet161_inference.xml --junit-prefix=inference_deepmark_densenet161_cpu"
    eval $cmd

    # 15. Run the densenet169
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_densenet169_inference.py --junit-xml=validation_test_deepmark_densenet169_inference.xml --junit-prefix=inference_deepmark_densenet169_cpu"
    eval $cmd

    # 16. Run the densenet201
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_densenet201_inference.py --junit-xml=validation_test_deepmark_densenet201_inference.xml --junit-prefix=inference_deepmark_densenet201_cpu"
    eval $cmd

    # 17. Run the squeezenet1.1
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_squeezenet_inference.py --junit-xml=validation_test_deepmark_squeezenet_inference.xml --junit-prefix=inference_deepmark_squeezenet_cpu"
    eval $cmd

    # 18. Run squeezenet1.0
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_squeezenet1_0_inference.py --junit-xml=validation_test_deepmark_squeezenet1_0_inference.xml --junit-prefix=inference_deepmark_squeezenet1_0_cpu"
    eval $cmd

    # 19. Run DCGAN 
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_dcgan_inference.py --junit-xml=validation_test_deepmark_dcgan_inference.xml --junit-prefix=inference_deepmark_dcgan_cpu"
    eval $cmd

    # 20. Run  sockeye_transformer
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_sockeye_transformer_inference.py --junit-xml=validation_test_deepmark_sockeye_transformer_inference.xml --junit-prefix=inference_deepmark_sockeye_transformer_cpu"
    eval $cmd

    # 21. Run  sockeye_gnmt
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_sockeye_gnmt_inference.py --junit-xml=validation_test_deepmark_sockeye_gnmt_inference.xml --junit-prefix=inference_deepmark_sockeye_gnmt_cpu"
    eval $cmd

    # 22. Run ssd_512_mobilenet1_0_voc 
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_ssd_512_mobilenet_inference.py --junit-xml=validation_test_deepmark_ssd_512_mobilenet_inference.xml --junit-prefix=inference_deepmark_ssd_512_mobilenet_cpu"
    eval $cmd

    # 23. Run ssd
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_ssd_inference.py --junit-xml=validation_test_deepmark_ssd_inference.xml --junit-prefix=inference_deepmark_ssd_cpu"
    eval $cmd

    # 24. Run vgg16
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_vgg16_inference.py --junit-xml=validation_test_deepmark_vgg16_inference.xml --junit-prefix=inference_deepmark_vgg16_cpu"
    eval $cmd

    # 25. Run deepspeech2_mod
    cmd="pytest -s ${INFERENCE_PY_SCRIPTS}test_deepmark_deepspeech2_mod_inference.py --junit-xml=validation_test_deepmark_deepspeech2_mod_inference.xml --junit-prefix=inference_deepmark_deepspeech2_mod_cpu"
    eval $cmd

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
