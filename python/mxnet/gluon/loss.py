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
# pylint: disable=arguments-differ
""" losses for training neural networks """
from __future__ import absolute_import
__all__ = ['Loss', 'L2Loss', 'L1Loss',
           'SigmoidBinaryCrossEntropyLoss', 'SigmoidBCELoss',
           'SoftmaxCrossEntropyLoss', 'SoftmaxCELoss',
           'KLDivLoss', 'CTCLoss']

from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock

def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss

def _reshape_label_as_output(F, output, label):
    # for symbolic output.shape is not available so we reshape
    # to empty shape and let it be inferred from output's shape
    # via the '-' operator later.
    return label.reshape(output.shape) if F is ndarray else label.reshape(())

class Loss(HybridBlock):
    """Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class L2Loss(Loss):
    """Calculates the mean squared error between output and label.

    .. math::
        L = \\frac{1}{2}\\sum_i \\Vert {output}_i - {label}_i \\Vert^2.

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Input shape:
        `output` is the prediction tensor.
        `label` is the truth tensor and should have the same shape as `output`.
        `sample_weight` is a matrix for per-sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10) and you want
        to weigh each sample in the batch separately, `sample_weight` should have shape (64, 1).

    Output shape:
        The loss output has the shape (batch_size,).
    """
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.square(output - label)
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class L1Loss(Loss):
    """Calculates the mean absolute error between output and label.

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    Output and label must have the same shape.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Input shape:
        `output` is the prediction tensor.
        `label` is the truth tensor and should have the same shape as `output`.
        `sample_weight` is a matrix for per-sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10) and you want
        to weigh each sample in the batch separately, `sample_weight` should have shape (64, 1).

    Output shape:
        The loss output has the shape (batch_size,).
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.abs(output - label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SigmoidBinaryCrossEntropyLoss(Loss):
    r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression.

    .. math::
        loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))


    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and then BCE, which is more numerically stable through
        log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Input shape:
        `output` is the prediction tensor.
        `label` is the truth tensor and should have the same shape as `output`.
        `sample_weight` is a matrix for per-sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10) and you want
        to weigh each sample in the batch separately, `sample_weight` should have shape (64, 1).

    Output shape:
        The loss output has the shape (batch_size,).
    """
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        if not self._from_sigmoid:
            max_val = F.maximum(-output, 0)
            loss = output - output*label + max_val + F.log(F.exp(-max_val)+F.exp(-output-max_val))
        else:
            loss = -(F.log(output+1e-12)*label + F.log(1.-output+1e-12)*(1.-label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

SigmoidBCELoss = SigmoidBinaryCrossEntropyLoss


class SoftmaxCrossEntropyLoss(Loss):
    """Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True`, label should contain integer category indicators:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i {log}(p_{i,{label}_i})

    Label's shape should be output's shape without the `axis` dimension. i.e. for
    `output.shape` = (1,2,3,4) and axis = 2, `label.shape` should be (1,2,4).

    If `sparse_label` is `False`, label should contain probability distribution
    with the same shape as output:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i \\sum_j {label}_j {log}(p_{ij})

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Input shape:
        `output` is the prediction tensor. The batch axis and softmax axis should be consistent
        with the value used in the constructor.
        `label` is the truth tensor. When `sparse_label` is true, `label` should have one less
        dimension (axis) than `output` tensor. Otherwise, the shape of `label` must be the same as
        output. For example, when `sparse_label` is true, if `output` has shape
        `(batch_size, x1, x2, c)` and axis is `-1`, `label` should have shape
        `(batch_size, x1, x2)`. If `sparse_label` is false, `label` should have shape
        `(batch_size, x1, x2, c)`.
        `sample_weight` is a matrix for per-sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10) and you want
        to weigh each sample in the batch separately, `sample_weight` should have shape (64, 1).

    Output shape:
        The loss output has the shape (batch_size,).
    """
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        if self._sparse_label:
            loss = -F.pick(output, label, axis=self._axis, keepdims=True)
        else:
            loss = -F.sum(output*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

SoftmaxCELoss = SoftmaxCrossEntropyLoss


class KLDivLoss(Loss):
    """The Kullback-Leibler divergence loss.

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    .. math::
        L = 1/n \\sum_i (label_i * (log(label_i) - output_i))

    Label's shape should be the same as output's.

    Parameters
    ----------
    from_logits : bool, default is `True`
        Whether the input is log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Input shape:
        `output` is the prediction tensor.
        `label` is the truth tensor and should have the same shape as `output`.
        `sample_weight` is a matrix for per-sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10) and you want
        to weigh each sample in the batch separately, `sample_weight` should have shape (64, 1).

    Output shape:
        The loss output has the shape (batch_size,).
    """
    def __init__(self, from_logits=True, weight=None, batch_axis=0, **kwargs):
        super(KLDivLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        loss = label * (F.log(label+1e-12) - output)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class CTCLoss(Loss):
    r"""Connectionist Temporal Classification Loss.

    See `"Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks"
    <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ paper for more information.

    Parameters
    ----------
    layout : str, default 'NTC'
        Layout of the output sequence activation vector.
    label_layout : str, default 'NT'
        Layout of the labels.
    weight : float or None
        Global scalar weight for loss.


    Input shape:
        `data` is an activation tensor (i.e. before softmax).
        Its shape depends on `layout`. For `layout='TNC'`, this
        input has shape `(sequence_length, batch_size, alphabet_size)`
        Note that the last dimension with index `alphabet_size-1` is reserved for special
        blank character.

        `label` is the label index matrix with zero-indexed labels.
        Its shape depends on `label_layout`. For `label_layout='TN'`, this
        input has shape `(label_sequence_length, batch_size)`. Padding mask of value ``-1``
        is available for dealing with unaligned label lengths.
        When `label_lengths` is specified, label lengths are directly used and padding mask
        is not allowed in the label.
        When `label_lengths` is not specified, the first occurrence of ``-1``
        in each sample marks the end of the label sequence of that sample.

        For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three
        sequences 'ba', 'cbb', and 'abac'. We can index the labels as `{'a': 0, 'b': 1, 'c': 2}`.
        The alphabet size should be 4, and we reserve the channel index 3 for blank label
        in data tensor. The padding mask value for extra length is -1, so the resulting `label`
        tensor should be padded to be::

          [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]

        `data_lengths` is optional and defaults to None.
        When specified, it represents the actual lengths of data.
        The shape should be (batch_size,).
        If None, the data lengths are treated as being equal to the max sequence length.
        This should be used as the third argument when calling this loss.

        `label_lengths` is optional and defaults to None.
        When specified, it represents the actual lengths of labels.
        The shape should be (batch_size,).
        If None, the label lengths are derived from the first occurrence of
        the value specified by `padding_mask`.
        This should be used as the fourth argument when calling this loss.

    Output shape:
        The CTC loss output has the shape (batch_size,).
    """
    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
               "Only 'NTC' and 'TNC' layouts for output are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
               "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, data, label,
                       data_lengths=None, label_lengths=None, sample_weight=None):
        if self._layout == 'NTC':
            data = F.swapaxes(data, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)
        loss = F.contrib.CTCLoss(data, label,
                                 use_data_lengths=data_lengths is not None,
                                 use_label_lengths=label_lengths is not None,
                                 data_lengths=data_lengths, label_lengths=label_lengths,
                                 blank_label='last')
        return _apply_weighting(F, loss, self._weight, sample_weight)
