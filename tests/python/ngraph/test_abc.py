# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# ----------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import mxnet as mx
import argparse
import sys



def test_abc():
    context = mx.cpu()
    if args.is_nnp:
        context = mx.nnp()
    A = mx.symbol.Variable('A')
    B = mx.symbol.Variable('B')
    C = mx.symbol.Variable('C')
    D = mx.symbol.Variable('D')
    Avals = np.array([[1,2],[3,4]], dtype=np.float32)
    Bvals = np.array([[5,6],[7,8]], dtype=np.float32)
    Cvals = np.array([[9,10],[11,12]], dtype=np.float32)
    Dvals = np.array([[0,0],[0,0]], dtype=np.float32)
    Aarr = mx.nd.array(Avals,ctx=context)
    Barr = mx.nd.array(Bvals,ctx=context)
    Carr = mx.nd.array(Cvals,ctx=context)
    Darr = mx.nd.array(Dvals,ctx=context)
    out1 = (A+B)*C + D
    exec1 = out1.bind(context,args=[Aarr, Barr,Carr, Darr])
    exec1.forward(is_train=False)
    out1 = exec1.outputs[0].asnumpy()
    assert (out1 == (Avals + Bvals) * Cvals+ Dvals).all()


    out2 = (B+C)*A + D
    exec2 = out2.bind(context, args=[Barr, Carr, Aarr, Darr])
    exec2.forward(is_train=False)
    out2 = exec2.outputs[0].asnumpy()
    assert (out2 == (Bvals + Cvals) * Avals+ Dvals).all()

    out3 = (C+A)*B + D
    exec3 = out3.bind(context,args=[Carr, Aarr, Barr, Darr])
    exec3.forward(is_train=False)
    out3 = exec3.outputs[0].asnumpy()
    assert (out3 == (Cvals + Avals) * Bvals + Dvals).all()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="unit test args", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--with-nnp',  action="store_true", default=False, dest="is_nnp")
    args = parser.parse_args()
    try:
        sys.argv.remove('--with-nnp')
    except ValueError:
        pass
    import nose
    nose.runmodule()
