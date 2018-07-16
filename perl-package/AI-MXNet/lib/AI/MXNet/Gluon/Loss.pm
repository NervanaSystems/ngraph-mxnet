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

use strict;
use warnings;
package AI::MXNet::Gluon::Loss;
use AI::MXNet::Gluon::Block;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Gluon::Loss - Base class for loss.
=cut

=head2 DESCRIPTION

    Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

=head2 _apply_weighting

    Apply weighting to loss.

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
=cut


method _apply_weighting(Str $F, GluonInput $loss, Maybe[Num] $weight=, Maybe[GluonInput] $sample_weight=)
{
    if(defined $sample_weight)
    {
        $loss = $F->broadcast_mul($loss, $sample_weight);
    }
    if(defined $weight)
    {
        $loss = $loss * $weight;
    }
    return $loss;
}

# Reshapes x to the same shape as y
method _reshape_like(GluonClass $F, GluonInput $x, GluonInput $y)
{
    if($F eq 'AI::MXNet::NDArray')
    {
        return $x->reshape($y->shape);
    }
    else
    {
        return $F->reshape_like($x, $y);
    }
}

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'weight'     => (is => 'rw', isa => 'Num');
has 'batch_axis' => (is => 'rw', isa => 'Int', default => 0);

use overload '""' => sub {
        my $self = shift;
        sprintf(
            "%s(batch_axis=%s, w=%s)",
            $self->_class_name,
            $self->batch_axis,
            $self->weight
        );
    };

method hybrid_forward($F, $x, @args)
{
    confess('NotImplementedError');
}

package AI::MXNet::Gluon::L2Loss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::L2Loss
=cut

=head1 DESCRIPTION

    Calculates the mean squared error between output and label:

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut
has '+weight'     => (default => 1);
has '+batch_axis' => (default => 0);

method hybrid_forward(GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{

    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss = $F->square($pred - $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight/2, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::L1Loss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has '+weight'     => (default => 1);
has '+batch_axis' => (default => 0);

=head1 NAME

    AI::MXNet::Gluon::L1Loss
=cut

=head1 DESCRIPTION

    Calculates the mean absolute error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    Output and label must have the same shape.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss = $F->abs($pred - $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'from_sigmoid' => (is => 'ro', isa => 'Bool', default => 0);
has '+batch_axis'  => (default => 0);

=head1 NAME

    AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss
=cut

=head1 DESCRIPTION

    The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

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
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss;
    if(not $self->from_sigmoid)
    {
        $loss = $F->relu($pred) - $pred * $label + $F->Activation(-$F->abs($pred), act_type=>'softrelu');
    }
    else
    {
        $loss = -($F->log($pred+1e-12)*$label + $F->log(1-$pred+1e-12)*(1-$label));
    }
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SigmoidBCELoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss';

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SoftmaxCrossEntropyLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';


=head1 NAME

    AI::MXNet::Gluon::SoftmaxCrossEntropyLoss
=cut

=head1 DESCRIPTION

    Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

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
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

has 'axis'         => (is => 'ro', isa => 'Int', default => -1);
has '+batch_axis'  => (default => 0);
has 'sparse_label' => (is => 'ro', isa => 'Bool', default => 1);
has 'from_logits'  => (is => 'ro', isa => 'Bool', default => 0);

method hybrid_forward(GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    if(not $self->from_logits)
    {
        $pred = $F->log_softmax($pred, axis => $self->axis);
    }
    my $loss;
    if($self->sparse_label)
    {
        $loss = -$F->pick($pred, $label, axis=>$self->axis, keepdims => 1);
    }
    else
    {
        __PACKAGE__->reshape_like($F, $label, $pred);
        $loss = -$F->sum($pred*$label, axis => $self->axis, keepdims => 1);
    }
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SoftmaxCELoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::SoftmaxCrossEntropyLoss';

__PACKAGE__->register('AI::MXNet::Gluon::Loss');


package AI::MXNet::Gluon::KLDivLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has '+batch_axis'  => (default => 0);
has 'axis'         => (is => 'ro', isa => 'Int', default => -1);
has 'from_logits'  => (is => 'ro', isa => 'Bool', default => 1);

=head1 NAME

    AI::MXNet::Gluon::KLDivLoss
=cut

=head1 DESCRIPTION

    The Kullback-Leibler divergence loss.

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
    axis : int, default -1
        The dimension along with to compute softmax. Only used when `from_logits`
        is False.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    if(not $self->from_logits)
    {
        $pred = $F->log_softmax($pred, axis => $self->axis);
    }
    my $loss = $label * ($F->log($label+1e-12) - $pred);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::CTCLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'layout'        => (is => 'rw', isa => 'Str', default => 'NTC');
has 'label_layout'  => (is => 'rw', isa => 'Str', default => 'NT');

=head1 NAME

    AI::MXNet::Gluon::CTCLoss
=cut

=head1 DESCRIPTION

    Connectionist Temporal Classification Loss.

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
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
        This should be used as the fifth argument when calling this loss.

    Input shapes:
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
=cut
use AI::MXNet::Base;

sub BUILD
{
    my $self = shift;
    assert(
        (grep { $_ eq $self->layout } ('NTC', 'TNC')),
        "Only 'NTC' and 'TNC' layouts for output are supported. Got: ${\ $self->layout }"
    );
    assert(
        (grep { $_ eq $self->label_layout } ('NT', 'TN')),
        "Only 'NT' and 'TN' layouts for label are supported. Got: ${\ $self->label_layout }"
    );
    $self->batch_axis(index($self->label_layout, 'N'));
}

method hybrid_forward(
    GluonClass $F, GluonInput $data, GluonInput $label,
    Maybe[GluonInput] $data_lengths=, Maybe[GluonInput] $label_lengths=, Maybe[GluonInput] $sample_weight=
)
{
    if($self->layout eq 'NTC')
    {
        $data = $F->swapaxes($data, dim1 => 0, dim2 => 1);
    }
    if($self->batch_axis == 1)
    {
        $label = $F->swapaxes($label, dim1 => 0, dim2 => 1);
    }
    my $loss = $F->contrib->CTCLoss(
        $data, $label,
        (defined $data_lengths ? $data_lengths : ()),
        (defined $label_lengths ? $label_lengths : ()),
        use_data_lengths  => defined $data_lengths ? 1 : 0,
        use_label_lengths => defined $label_lengths ? 1 : 0,
        blank_label=>'last'
    );
    return $self->_apply_weighting($F, $loss, $self->weight, $sample_weight);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::HuberLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'rho' => (is => 'rw', isa => 'Num', default => 1);

=head1 NAME

    AI::MXNet::Gluon::HuberLoss
=cut

=head1 DESCRIPTION

    Calculates smoothed L1 loss that is equal to L1 loss if absolute error
    exceeds rho but is equal to L2 loss otherwise. Also called SmoothedL1 loss.

    .. math::
        L = \sum_i \begin{cases} \frac{1}{2 {rho}} ({pred}_i - {label}_i)^2 &
                           \text{ if } |{pred}_i - {label}_i| < {rho} \\
                           |{pred}_i - {label}_i| - \frac{{rho}}{2} &
                           \text{ otherwise }
            \end{cases}

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    rho : float, default 1
        Threshold for trimmed mean estimator.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape [64, 10]
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape [64, 1].

    Outputs:
        - **loss**: loss tensor with shape [batch_size]. Dimenions other than
          batch_axis are averaged out.
=cut

method hybrid_forward(
    GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=
)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss = $F->abs($pred - $label);
    $loss = $F->where(
        $loss > $self->rho, $loss - 0.5 * $self->rho,
        (0.5/$self->rho) * $F->square($loss)
    );
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::HingeLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'margin' => (is => 'rw', isa => 'Num', default => 1);

=head1 NAME

    AI::MXNet::Gluon::HingeLoss
=cut

=head1 DESCRIPTION

    Calculates the hinge loss function often used in SVMs:

    .. math::
        L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1. `pred` and `label` must have the same number of
    elements.

    Parameters
    ----------
    margin : float
        The margin in hinge loss. Defaults to 1.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape.
        - **label**: truth tensor with values -1 or 1. Must have the same size
          as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
=cut

method hybrid_forward(
    GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=
)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss = $F->relu($self->margin - $pred * $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SquaredHingeLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'margin' => (is => 'rw', isa => 'Num', default => 1);

=head1 NAME

    AI::MXNet::Gluon::SquaredHingeLoss
=cut

=head1 DESCRIPTION

    Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)^2

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1. `pred` and `label` can have arbitrary shape as
    long as they have the same number of elements.

    Parameters
    ----------
    margin : float
        The margin in hinge loss. Defaults to 1.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: truth tensor with values -1 or 1. Must have the same size
          as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
=cut

method hybrid_forward(
    GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=
)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    my $loss = $F->square($F->relu($self->margin - $pred * $label));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::LogisticLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'label_format' => (is => 'rw', isa => 'Str', default => 'signed');

=head1 NAME

    AI::MXNet::Gluon::LogisticLoss
=cut

=head1 DESCRIPTION

    Calculates the logistic loss (for binary losses only):

    .. math::
        L = \sum_i \log(1 + \exp(- {pred}_i \cdot {label}_i))

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1 (0 or 1 if `label_format` is binary).
     `pred` and `label` can have arbitrary shape as long as they have the same number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    label_format : str, default 'signed'
        Can be either 'signed' or 'binary'. If the label_format is 'signed', all label values should
        be either -1 or 1. If the label_format is 'binary', all label values should be either
        0 or 1.

    Inputs:
        - **pred**: prediction tensor with arbitrary shape.
        - **label**: truth tensor with values -1/1 (label_format is 'signed')
          or 0/1 (label_format is 'binary'). Must have the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
=cut

sub BUILD
{
    my $self = shift;
    if(not ($self->label_format eq 'signed' or $self->label_format eq 'binary'))
    {
        confess(sprintf("label_format can only be signed or binary, recieved %s", $self->label_format));
    }
}

method hybrid_forward(
    GluonClass $F, GluonInput $pred, GluonInput $label, Maybe[GluonInput] $sample_weight=
)
{
    $label = __PACKAGE__->_reshape_like($F, $label, $pred);
    if($self->label_format eq 'signed')
    {
        $label = ($label + 1) / 2;  # Transform label to be either 0 or 1
    }
    # Use a stable formula in computation
    my $loss = $F->relu($pred) - $pred * $label + $F->Activation(-$F->abs($pred), act_type=>'softrelu');
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::TripletLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'margin' => (is => 'rw', isa => 'Num', default => 1);

=head1 NAME

    AI::MXNet::Gluon::TripletLoss
=cut

=head1 DESCRIPTION

    Calculates triplet loss given three input tensors and a positive margin.
    Triplet loss measures the relative similarity between prediction, a positive
    example and a negative example:

    .. math::
        L = \sum_i \max(\Vert {pred}_i - {pos_i} \Vert_2^2 -
                        \Vert {pred}_i - {neg_i} \Vert_2^2 + {margin}, 0)

    `pred`, `positive` and `negative` can have arbitrary shape as long as they
    have the same number of elements.

    Parameters
    ----------
    margin : float
        Margin of separation between correct and incorrect pair.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **positive**: positive example tensor with arbitrary shape. Must have
          the same size as pred.
        - **negative**: negative example tensor with arbitrary shape Must have
          the same size as pred.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
=cut

method hybrid_forward(
    GluonClass $F, GluonInput $pred, GluonInput $positive, GluonInput $negative, Maybe[GluonInput] $sample_weight=
)
{
    $positive = __PACKAGE__->_reshape_like($F, $positive, $pred);
    $negative = __PACKAGE__->_reshape_like($F, $negative, $pred);
    my $loss = $F->sum($F->square($pred-$positive) - $F->square($pred-$negative),
                     axis=>$self->batch_axis, exclude=>1);
    $loss = $F->relu($loss + $self->margin);
    return __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

1;