from __future__ import print_function
import numpy as np
import mxnet as mx

def test_device_lcr_enabled():
    with mx.nnp(0):
    	A = mx.symbol.Variable('A')
    	B = mx.symbol.Variable('B')
    	C = mx.symbol.Variable('C')
    	Avals = mx.nd.array([[1,2],[3,4]])
    	Bvals = mx.nd.array([[5,6],[7,8]])
    	Cvals = mx.nd.array([[9,10],[11,12]])
    	out = (A+B) * C
    	executor = out.bind(mx.nnp(0),args=[Avals,Bvals,Cvals])
    	#executor.forward(is_train=False)
   
    	for arr in executor.arg_arrays:
		assert arr.context == mx.nnp(0)
	

if __name__ == '__main__':
    import nose
    nose.runmodule()
