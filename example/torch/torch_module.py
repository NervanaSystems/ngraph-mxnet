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

# pylint: skip-file
import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
from get_data import MNISTIterator
import mxnet as mx
import numpy as np
import logging

# define mlp

use_torch_criterion = False

data = mx.symbol.Variable('data')
fc1 = mx.symbol.TorchModule(data_0=data, lua_string='nn.Linear(784, 128)', num_data=1, num_params=2, num_outputs=1, name='fc1')
act1 = mx.symbol.TorchModule(data_0=fc1, lua_string='nn.ReLU(false)', num_data=1, num_params=0, num_outputs=1, name='relu1')
fc2 = mx.symbol.TorchModule(data_0=act1, lua_string='nn.Linear(128, 64)', num_data=1, num_params=2, num_outputs=1, name='fc2')
act2 = mx.symbol.TorchModule(data_0=fc2, lua_string='nn.ReLU(false)', num_data=1, num_params=0, num_outputs=1, name='relu2')
fc3 = mx.symbol.TorchModule(data_0=act2, lua_string='nn.Linear(64, 10)', num_data=1, num_params=2, num_outputs=1, name='fc3')

if use_torch_criterion:
    logsoftmax = mx.symbol.TorchModule(data_0=fc3, lua_string='nn.LogSoftMax()', num_data=1, num_params=0, num_outputs=1, name='logsoftmax')
    # Torch's label starts from 1
    label = mx.symbol.Variable('softmax_label') + 1
    mlp = mx.symbol.TorchCriterion(data=logsoftmax, label=label, lua_string='nn.ClassNLLCriterion()', name='softmax')
else:
    mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

# data

train, val = MNISTIterator(batch_size=100, input_shape = (784,))

# train

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx = mx.cpu(0), symbol = mlp, num_epoch = 20,
    learning_rate = 0.1, momentum = 0.9, wd = 0.00001)

if use_torch_criterion:
    model.fit(X=train, eval_data=val, eval_metric=mx.metric.Torch())
else:
    model.fit(X=train, eval_data=val)
