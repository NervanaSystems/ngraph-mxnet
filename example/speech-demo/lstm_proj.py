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

# pylint:skip-file
import mxnet as mx
import numpy as np
from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias",
                                     "ph2h_weight",
                                     "c2i_bias", "c2f_bias", "c2o_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0., num_hidden_proj=0):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                #bias=param.h2h_bias,
                                no_bias=True,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))

    Wcidc = mx.sym.broadcast_mul(param.c2i_bias,  prev_state.c) + slice_gates[0]
    in_gate = mx.sym.Activation(Wcidc, act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")

    Wcfdc = mx.sym.broadcast_mul(param.c2f_bias, prev_state.c) + slice_gates[2]
    forget_gate = mx.sym.Activation(Wcfdc, act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)

    Wcoct = mx.sym.broadcast_mul(param.c2o_bias, next_c) + slice_gates[3]
    out_gate = mx.sym.Activation(Wcoct, act_type="sigmoid")

    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    if num_hidden_proj > 0:
        proj_next_h = mx.sym.FullyConnected(data=next_h,
                                            weight=param.ph2h_weight,
                                            no_bias=True,
                                            num_hidden=num_hidden_proj,
                                            name="t%d_l%d_ph2h" % (seqidx, layeridx))

        return LSTMState(c=next_c, h=proj_next_h)
    else:
        return LSTMState(c=next_c, h=next_h)

def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_label, dropout=0., output_states=False, take_softmax=True, num_hidden_proj=0):

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i),
                                     ph2h_weight = mx.sym.Variable("l%d_ph2h_weight" % i),
                                     c2i_bias = mx.sym.Variable("l%d_c2i_bias" % i, shape=(1,num_hidden)),
                                     c2f_bias = mx.sym.Variable("l%d_c2f_bias" % i, shape=(1,num_hidden)),
                                     c2o_bias = mx.sym.Variable("l%d_c2o_bias" % i, shape=(1, num_hidden))
                                     ))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    dataSlice = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = dataSlice[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp = 0.
            else:
                dp = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp, num_hidden_proj=num_hidden_proj)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=1)
    if num_hidden_proj > 0:
        hidden_final = mx.sym.Reshape(hidden_concat, target_shape=(0, num_hidden_proj))
    else:
        hidden_final = mx.sym.Reshape(hidden_concat, target_shape=(0, num_hidden))
    pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')
    pred = mx.sym.Reshape(pred, shape=(-1, num_label))
    label = mx.sym.Reshape(label, shape=(-1,))
    if take_softmax:
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, ignore_label=0,
                                  use_ignore=True, name='softmax')
    else:
        sm = pred

    if output_states:
        # block the gradients of output states
        for i in range(num_lstm_layer):
            state = last_states[i]
            state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                              h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
            last_states[i] = state

        # also output states, used in truncated-bptt to copy over states
        unpack_c = [state.c for state in last_states]
        unpack_h = [state.h for state in last_states]
        sm = mx.sym.Group([sm] + unpack_c + unpack_h)

    return sm
