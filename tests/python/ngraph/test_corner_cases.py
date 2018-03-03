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

from __future__ import print_function
import numpy as np
import mxnet as mx


def binary_op_ex(sym, x_shape, y_shape):
    np.random.seed(0)
    x_npy = np.random.randint(0, 10, size=x_shape).astype(np.float32)
    y_npy = np.random.randint(0, 10, size=y_shape).astype(np.float32)
    exe = sym.simple_bind(ctx=mx.cpu(), x=x_shape, y=y_shape)
    mx_out = exe.forward(is_train=True, x=x_npy, y=y_npy)[0].asnumpy()
    exe.backward()
    return mx_out


def _test_broadcast_op_no_head_grad():
    x = mx.symbol.Variable("x")
    y = mx.symbol.Variable("y")
    z = mx.sym.broadcast_not_equal(x, y)
    binary_op_ex(z, (1, 10), (10, 1))


def test_broadcast_mix_logic_op():
    x_shape = (1, 10)
    y_shape = (10, 1)
    x = mx.symbol.Variable("x")
    y = mx.symbol.Variable("y")
    z1 = mx.sym.broadcast_mul(x, y)
    z2 = mx.sym.broadcast_not_equal(z1, y)
    z3 = mx.sym.broadcast_mul(z1, z2)
    z4 = mx.sym.broadcast_equal(z1, z3)
    z5 = mx.sym.broadcast_not_equal(z3, z4)
    z6 = mx.sym.broadcast_mul(z5, z4)
    z = mx.sym.broadcast_equal(z6, x)

    binary_op_ex(z, (1, 10), (10, 1))

if __name__ == '__main__':
    import nose
    nose.runmodule()
