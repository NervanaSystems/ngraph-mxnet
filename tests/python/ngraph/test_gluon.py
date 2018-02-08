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
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

    def forward(self, x, y):
        return x * y


class NetHybrid(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NetHybrid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, y):
        # print('type(x): {}, F: {}'.format(
        #         type(x).__name__, F.__name__))
        return x * y


def test_gluon_basic():
    net = Net()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    # compute gluon forward
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)


def test_gluon_hybrid():
    net = NetHybrid()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    # compute gluon hybrid forward
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)


def test_gluon_hybridize():
    net = NetHybrid()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    # compute hybrid forward
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)
    # hybridize the network
    net.hybridize()
    # compute forward after hybridize
    z = net(a, b)
    assert_almost_equal(z.asnumpy(), y.asnumpy(), rtol=1e-3, atol=1e-6)


if __name__ == '__main__':
    import nose
    nose.runmodule()
