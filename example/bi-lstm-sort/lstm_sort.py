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

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import bi_lstm_unroll
from sort_io import BucketSentenceIter, default_build_vocab

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 100
    buckets = []
    num_hidden = 300
    num_embed = 512
    num_lstm_layer = 2

    num_epoch = 1
    learning_rate = 0.1
    momentum = 0.9

    contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab("./data/sort.train.txt")

    def sym_gen(seq_len):
        return bi_lstm_unroll(seq_len, len(vocab),
                              num_hidden=num_hidden, num_embed=num_embed,
                              num_label=len(vocab))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter("./data/sort.train.txt", vocab,
                                    buckets, batch_size, init_states)
    data_val = BucketSentenceIter("./data/sort.valid.txt", vocab,
                                  buckets, batch_size, init_states)

    if len(buckets) == 1:
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

    model.save("sort")
