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

# coding: utf-8
# pylint: disable= arguments-differ
"""Custom neural network layers in model_zoo."""
__all__ = ['Concurrent', 'HybridConcurrent', 'Identity', 'SparseEmbedding']

from .... import nd
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential

class Concurrent(Sequential):
    """Lays `Block`s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = Concurrent()
        # use net's name_scope to give children blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1, prefix=None, params=None):
        super(Concurrent, self).__init__(prefix=prefix, params=params)
        self.axis = axis

    def forward(self, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        out = nd.concat(*out, dim=self.axis)
        return out


class HybridConcurrent(HybridSequential):
    """Lays `HybridBlock`s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give children blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1, prefix=None, params=None):
        super(HybridConcurrent, self).__init__(prefix=prefix, params=params)
        self.axis = axis

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        out = F.concat(*out, dim=self.axis)
        return out


class Identity(HybridBlock):
    """Block that passes through the input directly.

    This block can be used in conjunction with HybridConcurrent
    block for residual connection.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())
    """
    def __init__(self, prefix=None, params=None):
        super(Identity, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        return x

class SparseEmbedding(Block):
    r"""Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]

    This SparseBlock is designed for distributed training with extremely large
    input dimension. Both weight and gradient w.r.t. weight are `RowSparseNDArray`.

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : Initializer
        Initializer for the `embeddings` matrix.

    Inputs:
        - **data**: (N-1)-D tensor with shape: `(x1, x2, ..., xN-1)`.
    Output:
        - **out**: N-D tensor with shape: `(x1, x2, ..., xN-1, output_dim)`.
    """
    def __init__(self, input_dim, output_dim, dtype='float32',
                 weight_initializer=None, **kwargs):
        super(SparseEmbedding, self).__init__(**kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype, 'sparse_grad': True}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=weight_initializer, dtype=dtype,
                                      grad_stype='row_sparse', stype='row_sparse')

    def forward(self, x):
        weight = self.weight.row_sparse_data(x)
        return nd.Embedding(x, weight, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__,
                        **self._kwargs)
