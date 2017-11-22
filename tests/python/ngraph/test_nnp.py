from __future__ import print_function
import numpy as np
import mxnet as mx

# def test_device_nnp_enabled():
#     with mx.nnp(0):
#         A = mx.symbol.Variable('A')
#         B = mx.symbol.Variable('B')
#         C = mx.symbol.Variable('C')
#         Avals = mx.nd.array([[1,2],[3,4]])
#         Bvals = mx.nd.array([[5,6],[7,8]])
#         Cvals = mx.nd.array([[9,10],[11,12]])
#         out = (A+B) * C
#         executor = out.bind(mx.nnp(0),args=[Avals,Bvals,Cvals])
#         #executor.forward(is_train=False)
#         
#         for arr in executor.arg_arrays:
#             assert arr.context == mx.nnp(0)
# 	
# 
# if __name__ == '__main__':
#     import nose
#     nose.runmodule()


def test_abc():
    A = mx.symbol.Variable('A')
    B = mx.symbol.Variable('B')
    C = mx.symbol.Variable('C')
    D = mx.symbol.Variable('D')
    Avals = np.array([[1,2],[3,4]], dtype=np.float32)
    Bvals = np.array([[5,6],[7,8]], dtype=np.float32)
    Cvals = np.array([[9,10],[11,12]], dtype=np.float32)
    Dvals = np.array([[0,0],[0,0]], dtype=np.float32)
    Aarr = mx.nd.array(Avals)
    Barr = mx.nd.array(Bvals)
    Carr = mx.nd.array(Cvals)
    Darr = mx.nd.array(Dvals)

    out1 = (A+B)*C + D
    exec1 = out1.bind(mx.nnp(0), args=[Aarr, Barr,Carr, Darr])
    exec1.forward(is_train=False)
    out1 = exec1.outputs[0].asnumpy()
    assert (out1 == (Avals + Bvals) * Cvals+ Dvals).all()


    out2 = (B+C)*A + D
    exec2 = out2.bind(mx.nnp(0),
                     args=[Barr, Carr, Aarr, Darr])
    exec2.forward(is_train=False)
    out2 = exec2.outputs[0].asnumpy()
    assert (out2 == (Bvals + Cvals) * Avals+ Dvals).all()

    out3 = (C+A)*B + D
    exec3 = out3.bind(mx.nnp(0), args=[Carr, Aarr, Barr, Darr])
    exec3.forward(is_train=False)
    out3 = exec3.outputs[0].asnumpy()
    assert (out3 == (Cvals + Avals) * Bvals + Dvals).all()


if __name__ == '__main__':
    import nose
    nose.runmodule()
