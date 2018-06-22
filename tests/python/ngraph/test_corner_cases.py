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


def test_broadcast_op_no_head_grad():
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

def run_softmax_forward_backward(sym, data):


    arr = [mx.nd.empty(data.shape)]
    arr_grad = [mx.nd.empty(data.shape)]
    arr[0][:] = data

    exec1 = sym.bind(mx.cpu(),
                   args=arr,
                   args_grad=arr_grad)

    mx_out = exec1.forward(is_train=True)
    mx_out = exec1.outputs[0].asnumpy()
    out_grad = mx.nd.ones(data.shape)

    exec1.backward([out_grad])

    return mx_out, arr_grad[0].asnumpy()

def test_softmax_activation():
    # Pre-defined random arrays and results from default mxnet CPU to compare with nGraph
    input1 = np.array([
       [0.53217715, 0.21630873, 0.9200532 , 0.44100983, 0.84649766],
       [0.12108511, 0.570474  , 0.27529801, 0.10380275, 0.81956027],
       [0.78868704, 0.52911125, 0.23378476, 0.23465436, 0.71470946],
       [0.1247181 , 0.54623195, 0.55646942, 0.24431403, 0.28930599],
       [0.32588183, 0.01083247, 0.44368248, 0.9458849 , 0.63206377]], dtype=np.float32
       )

    x = mx.symbol.Variable("x")
    z = mx.sym.SoftmaxActivation(x)
    forward, grad = run_softmax_forward_backward(z, input1)
    true_forward = np.array(
                 [[0.18230888, 0.1329315 , 0.2686954 , 0.1664234 , 0.24964076],
                  [0.14864044, 0.2329722 , 0.17342463, 0.14609365, 0.298869  ],
                  [0.2598194 , 0.20041914, 0.14916967, 0.14929944, 0.24129231],
                  [0.15698175, 0.23928213, 0.24174437, 0.17692491, 0.18506691],
                  [0.16469213, 0.12018456, 0.1852819 , 0.30615175, 0.22368969]], 
                  dtype=np.float32)
    true_backward = np.array(
                 [[ 1.08664562e-08,  7.92333488e-09,  1.60154947e-08,
                    9.91960736e-09,  1.48797490e-08],
                  [ 8.85966056e-09,  1.38862255e-08,  1.03369135e-08,
                    8.70786021e-09,  1.78139814e-08],
                  [ 1.54864424e-08,  1.19459118e-08,  8.89120511e-09,
                    8.89894025e-09,  1.43821426e-08],
                  [-1.87136830e-08, -2.85246529e-08, -2.88181745e-08,
                   -2.10910933e-08, -2.20616947e-08],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00]], dtype=np.float32)
    assert np.allclose(forward, true_forward)
    assert np.allclose(grad, true_backward, rtol=1e-3, atol=1e-7)

    input2 = np.array([
       [[0.17609602, 0.90801173, 0.44554609, 0.99507307, 0.84106854],
        [0.47028311, 0.78104274, 0.99057336, 0.11671654, 0.20584013],
        [0.51475785, 0.53206265, 0.46540811, 0.86673823, 0.99107314],
        [0.30808649, 0.66008214, 0.64497348, 0.89071153, 0.13196113],
        [0.16622791, 0.79404531, 0.31142678, 0.33842611, 0.42792043]],

       [[0.05108877, 0.96675625, 0.6077298 , 0.17584953, 0.40837491],
        [0.04824774, 0.81515839, 0.17412009, 0.79592174, 0.25124007],
        [0.42434969, 0.26389039, 0.08364128, 0.62277346, 0.64769867],
        [0.51270324, 0.22941933, 0.08571307, 0.06296123, 0.19530063],
        [0.08204256, 0.28028994, 0.84908637, 0.70048193, 0.39840395]],

       [[0.27865073, 0.55070251, 0.59278525, 0.7148701 , 0.97234936],
        [0.67327241, 0.10913394, 0.01542543, 0.30503719, 0.83647287],
        [0.13352289, 0.35623718, 0.07934908, 0.61714443, 0.73870447],
        [0.25548967, 0.16348307, 0.73962755, 0.26804754, 0.53329105],
        [0.85052903, 0.85473372, 0.24581613, 0.90716763, 0.80818399]],

       [[0.54995721, 0.00509445, 0.24617661, 0.23970231, 0.3423044 ],
        [0.92347134, 0.99974746, 0.07506943, 0.35241646, 0.471716  ],
        [0.66744447, 0.40374308, 0.36575651, 0.6874286 , 0.52490164],
        [0.31600314, 0.59628447, 0.91749988, 0.528848  , 0.25215523],
        [0.87566073, 0.92525822, 0.59844993, 0.53071467, 0.46937456]],

       [[0.40634484, 0.07199046, 0.41268365, 0.13759053, 0.26770341],
        [0.56530379, 0.56539445, 0.20004284, 0.29760379, 0.15485124],
        [0.55641568, 0.49869577, 0.96862646, 0.63867995, 0.13234088],
        [0.88355243, 0.51987049, 0.10487715, 0.07591599, 0.81723034],
        [0.17935192, 0.67845026, 0.5149324 , 0.77334196, 0.83712913]]], 
        dtype=np.float32)

    x = mx.symbol.Variable("x")
    z = mx.sym.SoftmaxActivation(x)
    forward, grad = run_softmax_forward_backward(z, input2)
    true_forward = np.array(
      [[[0.11626664, 0.24172528, 0.15222143, 0.26371348, 0.22607318],
        [0.18126483, 0.24732885, 0.304981  , 0.12728041, 0.13914494],
        [0.16663015, 0.16953874, 0.1586066 , 0.2369282 , 0.2682963 ],
        [0.15499721, 0.22039092, 0.2170861 , 0.27755862, 0.12996715],
        [0.15344891, 0.28748915, 0.1774283 , 0.18228401, 0.19934963]],

       [[0.1281137 , 0.3200847 , 0.22353303, 0.14513712, 0.18313147],
        [0.1311581 , 0.28239706, 0.14875129, 0.2770166 , 0.16067694],
        [0.19869834, 0.1692418 , 0.14132744, 0.24230848, 0.24842395],
        [0.265139  , 0.1997308 , 0.17299525, 0.16910373, 0.19303118],
        [0.13156492, 0.16041236, 0.28331068, 0.24418832, 0.18052365]],

       [[0.1383187 , 0.18156473, 0.18936853, 0.21395802, 0.27679   ],
        [0.25277787, 0.1437927 , 0.13093017, 0.17491075, 0.29758847],
        [0.1503702 , 0.1878821 , 0.1424408 , 0.24389109, 0.27541578],
        [0.1704235 , 0.15544313, 0.27655905, 0.17257716, 0.2249972 ],
        [0.2189498 , 0.21987234, 0.11959721, 0.23170872, 0.20987192]],

       [[0.25881973, 0.15009509, 0.19101484, 0.18978216, 0.2102882 ],
        [0.26962912, 0.29100007, 0.11542782, 0.15232135, 0.17162158],
        [0.22752573, 0.17478591, 0.16827093, 0.23211835, 0.19729902],
        [0.1581355 , 0.20929268, 0.2885733 , 0.1956441 , 0.14835447],
        [0.23905596, 0.2512115 , 0.18117926, 0.16931345, 0.15923984]],

       [[0.22951414, 0.16428624, 0.23097359, 0.17542477, 0.19980127],
        [0.24256574, 0.24258773, 0.1683444 , 0.1855961 , 0.16090597],
        [0.19252302, 0.18172522, 0.29073915, 0.20903045, 0.12598222],
        [0.28281614, 0.19658898, 0.12981647, 0.12611076, 0.26466766],
        [0.12840787, 0.21151797, 0.17961076, 0.23257242, 0.24789092]]],
      dtype=np.float32)
    assert np.allclose(forward, true_forward)
    # Note(mbrookhart): We don't test backward for this case, 
    # because MXNet's op returns the wrong values. 
    # In particular, mxnet does bprop assuming that the shapes 
    # are always compatible with mode="channel", which works for 2D softmax, 
    # but not for 3D softmax. 
    # We wont fix the op because it's deprecated, and all of the models 
    # that use SoftmaxActivation use the mode="channel" version 

    x = mx.symbol.Variable("x")
    z = mx.sym.SoftmaxActivation(x, mode="channel")
    forward, grad = run_softmax_forward_backward(z, input2)
    true_forward = np.array(
      [[[0.17017275, 0.23584345, 0.17123565, 0.26921445, 0.26005763],
        [0.22837777, 0.2077217 , 0.29532248, 0.11184923, 0.13778229],
        [0.23876408, 0.1619389 , 0.17467074, 0.23678997, 0.30214524],
        [0.19418368, 0.18405573, 0.20902796, 0.24253519, 0.12797   ],
        [0.16850172, 0.21044026, 0.14974314, 0.1396112 , 0.17204483]],

       [[0.16480716, 0.299739  , 0.24351296, 0.14274202, 0.20312186],
        [0.16433959, 0.25757587, 0.1578366 , 0.26536632, 0.17358565],
        [0.23937634, 0.14842004, 0.14418276, 0.2231765 , 0.25804392],
        [0.26148856, 0.14339101, 0.14448178, 0.12750438, 0.16414197],
        [0.16998833, 0.15087412, 0.30998588, 0.24121083, 0.2011066 ]],

       [[0.16400349, 0.22219482, 0.24845879, 0.22612426, 0.24050958],
        [0.24335212, 0.14287727, 0.13947943, 0.1500925 , 0.20995295],
        [0.14184855, 0.18292738, 0.1486866 , 0.20507155, 0.19039771],
        [0.16024865, 0.15085742, 0.2877578 , 0.14464204, 0.15504287],
        [0.2905472 , 0.30114314, 0.1756174 , 0.27406964, 0.20409685]],

       [[0.17379513, 0.10515874, 0.15750816, 0.15730369, 0.18560518],
        [0.25249496, 0.2843267 , 0.13273707, 0.17607191, 0.21124811],
        [0.1954617 , 0.15666655, 0.17751539, 0.24614102, 0.22278762],
        [0.13754115, 0.18993102, 0.30821595, 0.21004546, 0.16960506],
        [0.24070707, 0.26391703, 0.22402345, 0.21043792, 0.21075407]],

       [[0.17413409, 0.13211782, 0.1853655 , 0.1503155 , 0.15958211],
        [0.20413563, 0.21639341, 0.1498581 , 0.1763992 , 0.14255193],
        [0.2023293 , 0.20243107, 0.32320035, 0.2480985 , 0.13937888],
        [0.28062895, 0.2067632 , 0.13625431, 0.14132494, 0.27646536],
        [0.13877207, 0.2422945 , 0.20532177, 0.28386188, 0.28202176]]],
      dtype=np.float32)
    true_backward= np.array(
      [[[ 0.00000000e+00,  0.00000000e+00,  1.02064401e-08,
          0.00000000e+00,  1.55006425e-08],
        [ 0.00000000e+00,  0.00000000e+00,  1.76025914e-08,
          0.00000000e+00,  8.21246449e-09],
        [ 0.00000000e+00,  0.00000000e+00,  1.04111875e-08,
          0.00000000e+00,  1.80092599e-08],
        [ 0.00000000e+00,  0.00000000e+00,  1.24590374e-08,
          0.00000000e+00,  7.62760610e-09],
        [ 0.00000000e+00,  0.00000000e+00,  8.92538665e-09,
          0.00000000e+00,  1.02546709e-08]],

       [[ 9.82327197e-09,  0.00000000e+00,  1.45145034e-08,
          0.00000000e+00,  0.00000000e+00],
        [ 9.79540271e-09,  0.00000000e+00,  9.40779454e-09,
          0.00000000e+00,  0.00000000e+00],
        [ 1.42679415e-08,  0.00000000e+00,  8.59396199e-09,
          0.00000000e+00,  0.00000000e+00],
        [ 1.55859325e-08,  0.00000000e+00,  8.61178506e-09,
          0.00000000e+00,  0.00000000e+00],
        [ 1.01320943e-08,  0.00000000e+00,  1.84765980e-08,
          0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.43354884e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.25141710e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.13485878e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  9.24127530e-09],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.21651205e-08]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00, -2.21258620e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00, -2.51827377e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00, -2.65583537e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00, -2.02184989e-08],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00, -2.51238426e-08]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]]], dtype=np.float32)
    assert np.allclose(forward, true_forward)
    assert np.allclose(grad, true_backward, rtol=1e-3, atol=1e-7)

def test_batch_normalized_softmax_grad():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2],[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, 
                               use_ignore=False, normalization="batch")
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052], 
                            [0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((2,4))
    expected_grad_out[:, k] = - 1
    assert np.isclose(grad_out , (expected_softmax_out + expected_grad_out) / 2).all()

def test_valid_normalized_softmax_grad():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2],[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, 
                               use_ignore=True, normalization="valid")
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052], 
                            [0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((2,4))
    expected_grad_out[:, k] = - 1
    
    assert np.isclose(grad_out, (expected_softmax_out + expected_grad_out) 
                                 / sum(label_nd.asnumpy() != 0)).all()

def test_valid_make_loss():
    xpu = mx.cpu()
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[0, 1, 1, 0], 
                        [1, 1, 1, .1]], ctx=xpu)
    grad_x = mx.nd.zeros((2,4), ctx=xpu)
    label_nd = mx.nd.array([1,1], ctx=xpu)

    sym = mx.sym.MakeLoss(x, normalization="valid", valid_thresh=0.2)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, 
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    out = ex.outputs[0].asnumpy()
    expected_out = [[0, 1, 1, 0], 
                    [1, 1, 1, .1]]
    assert np.isclose(out, expected_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    expected_grad_out = np.ones((2,4))/5
    
    assert np.isclose(grad_out, expected_grad_out).all() 

def test_stop_gradient():                                    
    v1 = mx.nd.array([[1, 2]])                                 
    v2 = mx.nd.array([[0, 1]])                                 
    a = mx.sym.Variable('a')                                   
    b = mx.sym.Variable('b')                                   
    b_stop_grad = mx.sym.stop_gradient(3 * b)                  
    loss = mx.sym.MakeLoss(b_stop_grad + a)                    
                                                               
    executor = loss.simple_bind(ctx=mx.cpu(), a=(1,2), b=(1,2))
    executor.forward(is_train=True, a=v1, b=v2)                     
    assert np.isclose(executor.outputs[0].asnumpy(), [1,5]).all()
    executor.backward()                                  
    assert np.isclose(executor.grad_arrays[0].asnumpy(), [0,0]).all()
    assert np.isclose(executor.grad_arrays[1].asnumpy(), [1,1]).all()

if __name__ == '__main__':
    import nose
    nose.runmodule()
