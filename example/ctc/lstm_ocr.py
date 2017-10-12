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
from __future__ import print_function
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll

from captcha.image import ImageCaptcha
import cv2, random


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


def gen_rand():
    buf = ""
    max_len = random.randint(3, 4)
    for i in range(max_len):
        buf += str(random.randint(0, 9))
    return buf


def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret


class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, init_states):
        super(OCRIter, self).__init__()
        global SEQ_LENGTH
        # you can get this font from http://font.ubuntu.com/
        self.captcha = ImageCaptcha(fonts=['./data/Xerox.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 80, 30))] + init_states
        self.provide_label = [('label', (self.batch_size, 4))]
        self.cache_data = []
        self.cache_label = []

    def __iter__(self):
        print('iter')
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                num = gen_rand()
                img = self.captcha.generate(num)
                img = np.fromstring(img.getvalue(), dtype='uint8')
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (80, 30))
                img = img.transpose(1, 0)
                img = img.reshape((80, 30))
                img = np.multiply(img, 1 / 255.0)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        self.cache_data.clear()
        self.cache_label.clear()
        pass


BATCH_SIZE = 1024
SEQ_LENGTH = 80


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    rp = np.argmax(pred, axis=1)
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total


def LCS(p, l):
    # Dynamic Programming Finding LCS
    if len(p) == 0:
        return 0
    P = np.array(list(p)).reshape((1, len(p)))
    L = np.array(list(l)).reshape((len(l), 1))
    M = np.int32(P == L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i == 0 else M[i - 1, j]
            left = 0 if j == 0 else M[i, j - 1]
            M[i, j] = max(up, left, M[i, j] if (i == 0 or j == 0) else M[i, j] + M[i - 1, j - 1])
    return M.max()


def Accuracy_LCS(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        hit += LCS(p, l) * 1.0 / len(l)
        total += 1.0
    return hit / total


def asum_stat(x):
    """returns |x|/size(x), async execution."""
    # npx = x.asnumpy()
    # print(npx)
    return x
    return mx.ndarray.norm(x) / np.sqrt(x.size)


if __name__ == '__main__':
    num_hidden = 100
    num_lstm_layer = 2

    num_epoch = 100
    learning_rate = 0.01
    momentum = 0.9
    num_label = 4

    contexts = [mx.context.gpu(0)]


    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden=num_hidden,
                           num_label=num_label)


    init_c = [('l%d_init_c' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = OCRIter(20000, BATCH_SIZE, num_label, init_states)
    data_val = OCRIter(1000, BATCH_SIZE, num_label, init_states)

    symbol = sym_gen(SEQ_LENGTH)

    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print('begin fit')

    module = mx.mod.Module(symbol, data_names=['data', 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
                           label_names=['label'],
                           context=contexts)

    module.fit(train_data=data_train,
               eval_data=data_val,
               eval_metric=mx.metric.np(Accuracy, allow_extra_outputs=True),
               optimizer='sgd',
               optimizer_params={'learning_rate': learning_rate,
                                 'momentum': momentum,
                                 'wd': 0.00001,
                                 },
               initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
               num_epoch=num_epoch,
               batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 50),
               epoch_end_callback=mx.callback.do_checkpoint("ocr"),
               )
