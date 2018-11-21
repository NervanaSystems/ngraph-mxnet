#!/bin/bash
set -e
set -u

#*******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

# This script generates a set of nGraph-enabled MXnet wheels for Ubuntu 16.04.
#
# These wheels should NOT be confused with the official MXnet wheels published
# on `https://pypi.org/project/mxnet/`. Those wheels are generated and published
# by their respective PyPI maintainers.

declare THIS_SCRIPT_DIR="$(readlink -e "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"

function die {
    declare LINE
    for LINE in "${*}"; do
        echo "${LINE}" >&2
    done
    exit 1
}

declare MXNET_BUILD_DIR="$(readlink -e "${THIS_SCRIPT_DIR}"/..)"
declare WHEEL_OUTPUT_DIR="${THIS_SCRIPT_DIR}/wheels-ubuntu1604"

echo "THIS_SCRIPT_DIR='${THIS_SCRIPT_DIR}'"
echo "MXNET_BUILD_DIR='${MXNET_BUILD_DIR}'"
echo "WHEEL_OUTPUT_DIR='${WHEEL_OUTPUT_DIR}'"

