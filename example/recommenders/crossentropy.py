#!/usr/bin/env python

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

"""Cross-entropy loss layer for MXNet.
"""
import os

import numpy as np
import mxnet as mx

# ref: http://mxnet.io/how_to/new_op.html

class CrossEntropyLoss(mx.operator.CustomOp):
    """An output layer that calculates gradient for cross-entropy loss
    y * log(p) + (1-y) * log(p)
    for label "y" and prediction "p".
    However, the output of this layer is the original prediction -- same as
    the "data" input, making it useful for tasks like "predict".
    If you actually want to use the calculated loss, see CrossEntropyLoss op.

    This is useful for multi-label prediction where each possible output
    label is considered independently.
    Cross-entropy loss provides a very large penalty for guessing
    the wrong answer (0 or 1) confidently.
    The gradient calculation is optimized for y only being 0 or 1.
    """

    eps = 1e-6 # Avoid -inf when taking log(0)
    eps1 = 1. + eps
    eps_1 = 1. - eps

    def forward(self, is_train, req, in_data, out_data, aux):
        # Shapes:
        #  b = minibatch size
        #  d = number of dimensions
        actually_calculate_loss = False
        if actually_calculate_loss:
            p = in_data[0].asnumpy()  # shape=(b,d)
            y = in_data[1].asnumpy()
            out = y * np.log(p+self.eps) + (1.-y) * np.log((self.eps1) - p)
            self.assign(out_data[0], req[0], mx.nd.array(out))
        else:
            # Just copy the predictions forward
            self.assign(out_data[0], req[0], in_data[0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.approx_backward(req, out_grad, in_data, out_data, in_grad, aux)
        #self.exact_backward(req, out_grad, in_data, out_data, in_grad, aux)

    def approx_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Correct grad = (y-p)/(p-p^2)
        But if y is just 1 or 0, then this simplifies to
        grad = 1/(p-1+y)
        which is more numerically stable
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()
        grad = -1. / (p - self.eps_1 + y)
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


    def exact_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """grad = (y-p)/(p-p^2)
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()  # seems right
        grad = (p - y) / ((p+self.eps) * (self.eps1 - p))
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("CrossEntropyLoss")
class CrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['preds']

    def create_operator(self, ctx, shapes, dtypes):
        return CrossEntropyLoss()

    def infer_shape(self, in_shape):
        if in_shape[0] != in_shape[1]:
            raise ValueError("Input shapes differ. data:%s. label:%s. must be same"
                    % (str(in_shape[0]),str(in_shape[1])))
        output_shape = in_shape[0]
        return in_shape, [output_shape], []


if __name__ == "__main__":
    print("Simple test of cross-entropy")
    data = mx.symbol.Variable('data')
    labs = mx.symbol.Variable('labs')
    net = mx.symbol.Custom(data=data, label=labs, name='ce',
            op_type='CrossEntropyLoss')
    rand = np.random.RandomState(seed=123)
    for i in range(20):
        sz = (6,4)
        d = mx.nd.array(rand.uniform(0.01,0.99,sz))
        l = mx.nd.array(rand.randint(0,2,sz))
        e = net.bind(ctx=mx.cpu(), args={'data':d, 'labs':l})
        e.forward()
        print("D:%s" % d.asnumpy())
        print("L:%s" % l.asnumpy())
        print("out:%s" % e.outputs[0].asnumpy())
        out = e.outputs[0].asnumpy()
        if np.abs(out).max() > 1e20:
            raise ValueError("output too high!")
    print("Done with test")

