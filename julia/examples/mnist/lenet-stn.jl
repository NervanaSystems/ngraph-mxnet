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

using MXNet

#--------------------------------------------------------------------------------
# define lenet with stn layer



# input
data = mx.Variable(:data)


# the localisation network in lenet-stn
# it will increase acc about more than 1%, when num-epoch >=15
# The localization net just takes the data as input and must output a vector in R^n
loc_net = @mx.chain mx.Convolution(data, num_filter=10, kernel=(5, 5), stride=(2,2)) =>
                    mx.Activation(act_type=:relu) =>
                    mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>
                    mx.Convolution( num_filter=10, kernel=(3, 3), stride=(2,2), pad=(1, 1)) =>
                    mx.Activation(act_type=:relu) =>
                    mx.Pooling( global_pool=true, kernel=(2, 2), pool_type=:avg) =>
                    mx.Flatten() =>
                    mx.FullyConnected(num_hidden=6, name=:stn_loc)

data=mx.SpatialTransformer(data,loc_net, target_shape = (28,28), transform_type="affine", sampler_type="bilinear")

# first conv
conv1 = @mx.chain mx.Convolution(data, kernel=(5,5), num_filter=20)  =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

# second conv
conv2 = @mx.chain mx.Convolution(conv1, kernel=(5,5), num_filter=50) =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

# first fully-connected
fc1   = @mx.chain mx.Flatten(conv2) =>
                  mx.FullyConnected(num_hidden=500) =>
                  mx.Activation(act_type=:tanh)

# second fully-connected
fc2   = mx.FullyConnected(fc1, num_hidden=10)

# softmax loss
lenet = mx.SoftmaxOutput(fc2, name=:softmax)


#--------------------------------------------------------------------------------

# load data
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)

#--------------------------------------------------------------------------------
# fit model
model = mx.FeedForward(lenet, context=mx.cpu())

# optimizer
optimizer = mx.ADAM(η=0.01, λ=0.00001)

# fit parameters
initializer=mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 1)
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider,initializer=initializer)
