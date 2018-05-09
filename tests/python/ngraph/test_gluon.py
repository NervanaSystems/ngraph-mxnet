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
"""
ngraph-mxnet imperative gluon python unittests
"""

from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.test_utils import assert_almost_equal


class Net(gluon.Block):
    """
    sample gluon net
    """
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

    def forward(self, x, y):
        return x * y


class NetHybrid(gluon.HybridBlock):
    """
    sample gluon hybrid net
    """
    def __init__(self, **kwargs):
        super(NetHybrid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, y):
        # print('type(x): {}, F: {}'.format(
        #         type(x).__name__, F.__name__))
        return x * y


def test_gluon_basic():
    """
    simple gluon test
    """
    net = Net()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    # compute gluon forward
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)


def test_gluon_hybrid():
    """
    simple gluon hybrid test
    """
    net = NetHybrid()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    # compute gluon hybrid forward
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)


def test_gluon_hybridize():
    """
    simple gluon hybridize test
    """
    net = NetHybrid()
    a = mx.nd.array([1, 2])
    b = mx.nd.array([2, 3])
    delta = mx.nd.array([1])
    # compute hybrid forwarard
    y = net(a, b)
    assert_almost_equal(y.asnumpy(), np.array([2.0, 6.0]), rtol=1e-3, atol=1e-6)
    # hybridize the network
    net.hybridize()
    # compute forward after hybridize
    z = net(a, b)
    assert_almost_equal(z.asnumpy(), y.asnumpy(), rtol=1e-3, atol=1e-6)

    with autograd.record():
        a.attach_grad()
        b.attach_grad()
        output = net(a,b)
        output.backward(delta)

    assert_almost_equal(a.grad.asnumpy(), np.array([2,0]), rtol=1e-3, atol=1e-6)
    assert_almost_equal(b.grad.asnumpy(), np.array([1,0]), rtol=1e-3, atol=1e-6)

def test_ngraph_imperative_gluon_convolution():
    """
    This will test Gluon convolution op with ngraph imperative involving mkldnn layout
    """

    ctx = mx.cpu()
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.collect_params().initialize(ctx=ctx)
    x = mx.nd.array(np.ones([32, 3, 224, 224]), ctx)
    y = net(x)

    # trigger computation on ndarray slice
    assert_almost_equal(y[0].asnumpy()[0, 0, 0], 0.3376348)


if __name__ == '__main__':
    import nose
    nose.runmodule()
