#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e

MXNET_ROOT=$(cd "$(dirname $0)/../../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/osx-x86_64-cpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*:$MXNET_ROOT/scala-package/infer/target/*

# model dir
MODEL_PATH_PREFIX=$1
# input image
INPUT_IMG=$2
# which input image dir
INPUT_DIR=$3

java -Xmx8G -Dmxnet.traceLeakedObjects=true -cp $CLASS_PATH \
	org.apache.mxnetexamples.infer.imageclassifier.ImageClassifierExample \
	--model-path-prefix $MODEL_PATH_PREFIX \
	--input-image $INPUT_IMG \
	--input-dir $INPUT_DIR
