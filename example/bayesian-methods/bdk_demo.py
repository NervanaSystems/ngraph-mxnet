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

from __future__ import print_function
import mxnet as mx
import mxnet.ndarray as nd
import numpy
import logging
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
from algos import *
from data_loader import *
from utils import *


class CrossEntropySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(CrossEntropySoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1))).astype('float32')
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = (y - l)


class LogSoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(LogSoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = (x - x.max(axis=1, keepdims=True)).astype('float32')
        y -= numpy.log(numpy.exp(y).sum(axis=1, keepdims=True)).astype('float32')
        # y[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        # y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = (numpy.exp(y) - l).astype('float32')


def classification_student_grad(student_outputs, teacher_pred):
    return [student_outputs[0] - teacher_pred]


def regression_student_grad(student_outputs, teacher_pred, teacher_noise_precision):
    student_mean = student_outputs[0]
    student_var = student_outputs[1]
    grad_mean = nd.exp(-student_var) * (student_mean - teacher_pred)

    grad_var = (1 - nd.exp(-student_var) * (nd.square(student_mean - teacher_pred)
                                            + 1.0 / teacher_noise_precision)) / 2
    return [grad_mean, grad_var]


def get_mnist_sym(output_op=None, num_hidden=400):
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc1', num_hidden=num_hidden)
    net = mx.symbol.Activation(data=net, name='mnist_relu1', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc2', num_hidden=num_hidden)
    net = mx.symbol.Activation(data=net, name='mnist_relu2', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc3', num_hidden=10)
    if output_op is None:
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    else:
        net = output_op(data=net, name='softmax')
    return net


def synthetic_grad(X, theta, sigma1, sigma2, sigmax, rescale_grad=1.0, grad=None):
    if grad is None:
        grad = nd.empty(theta.shape, theta.context)
    theta1 = theta.asnumpy()[0]
    theta2 = theta.asnumpy()[1]
    v1 = sigma1 ** 2
    v2 = sigma2 ** 2
    vx = sigmax ** 2
    denominator = numpy.exp(-(X - theta1) ** 2 / (2 * vx)) + numpy.exp(
        -(X - theta1 - theta2) ** 2 / (2 * vx))
    grad_npy = numpy.zeros(theta.shape)
    grad_npy[0] = -rescale_grad * ((numpy.exp(-(X - theta1) ** 2 / (2 * vx)) * (X - theta1) / vx
                                    + numpy.exp(-(X - theta1 - theta2) ** 2 / (2 * vx)) * (
                                    X - theta1 - theta2) / vx) / denominator).sum() \
                  + theta1 / v1
    grad_npy[1] = -rescale_grad * ((numpy.exp(-(X - theta1 - theta2) ** 2 / (2 * vx)) * (
    X - theta1 - theta2) / vx) / denominator).sum() \
                  + theta2 / v2
    grad[:] = grad_npy
    return grad


def get_toy_sym(teacher=True, teacher_noise_precision=None):
    if teacher:
        net = mx.symbol.Variable('data')
        net = mx.symbol.FullyConnected(data=net, name='teacher_fc1', num_hidden=100)
        net = mx.symbol.Activation(data=net, name='teacher_relu1', act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name='teacher_fc2', num_hidden=1)
        net = mx.symbol.LinearRegressionOutput(data=net, name='teacher_output',
                                               grad_scale=teacher_noise_precision)
    else:
        net = mx.symbol.Variable('data')
        net = mx.symbol.FullyConnected(data=net, name='student_fc1', num_hidden=100)
        net = mx.symbol.Activation(data=net, name='student_relu1', act_type="relu")
        student_mean = mx.symbol.FullyConnected(data=net, name='student_mean', num_hidden=1)
        student_var = mx.symbol.FullyConnected(data=net, name='student_var', num_hidden=1)
        net = mx.symbol.Group([student_mean, student_var])
    return net


def dev():
    return mx.gpu()


def run_mnist_SGD(training_num=50000):
    X, Y, X_test, Y_test = load_mnist(training_num)
    minibatch_size = 100
    net = get_mnist_sym()
    data_shape = (minibatch_size,) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    exe, exe_params, _ = SGD(sym=net, dev=dev(), data_inputs=data_inputs, X=X, Y=Y,
                             X_test=X_test, Y_test=Y_test,
                             total_iter_num=1000000,
                             initializer=initializer,
                             lr=5E-6, prior_precision=1.0, minibatch_size=100)


def run_mnist_SGLD(training_num=50000):
    X, Y, X_test, Y_test = load_mnist(training_num)
    minibatch_size = 100
    net = get_mnist_sym()
    data_shape = (minibatch_size,) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    exe, sample_pool = SGLD(sym=net, dev=dev(), data_inputs=data_inputs, X=X, Y=Y,
                            X_test=X_test, Y_test=Y_test,
                            total_iter_num=1000000,
                            initializer=initializer,
                            learning_rate=4E-6, prior_precision=1.0, minibatch_size=100,
                            thin_interval=100, burn_in_iter_num=1000)


def run_mnist_DistilledSGLD(training_num=50000):
    X, Y, X_test, Y_test = load_mnist(training_num)
    minibatch_size = 100
    if training_num >= 10000:
        num_hidden = 800
        total_iter_num = 1000000
        teacher_learning_rate = 1E-6
        student_learning_rate = 0.0001
        teacher_prior = 1
        student_prior = 0.1
        perturb_deviation = 0.1
    else:
        num_hidden = 400
        total_iter_num = 20000
        teacher_learning_rate = 4E-5
        student_learning_rate = 0.0001
        teacher_prior = 1
        student_prior = 0.1
        perturb_deviation = 0.001
    teacher_net = get_mnist_sym(num_hidden=num_hidden)
    logsoftmax = LogSoftmax()
    student_net = get_mnist_sym(output_op=logsoftmax, num_hidden=num_hidden)
    data_shape = (minibatch_size,) + X.shape[1::]
    teacher_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                           'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                           'softmax_label': nd.zeros((minibatch_size, 10), ctx=dev())}
    teacher_initializer = BiasXavier(factor_type="in", magnitude=1)
    student_initializer = BiasXavier(factor_type="in", magnitude=1)
    student_exe, student_params, _ = \
        DistilledSGLD(teacher_sym=teacher_net, student_sym=student_net,
                      teacher_data_inputs=teacher_data_inputs,
                      student_data_inputs=student_data_inputs,
                      X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=total_iter_num,
                      student_initializer=student_initializer,
                      teacher_initializer=teacher_initializer,
                      student_optimizing_algorithm="adam",
                      teacher_learning_rate=teacher_learning_rate,
                      student_learning_rate=student_learning_rate,
                      teacher_prior_precision=teacher_prior, student_prior_precision=student_prior,
                      perturb_deviation=perturb_deviation, minibatch_size=100, dev=dev())


def run_toy_SGLD():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = 1
    teacher_noise_precision = 1.0 / 9.0
    net = get_toy_sym(True, teacher_noise_precision)
    data_shape = (minibatch_size,) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'teacher_output_label': nd.zeros((minibatch_size, 1), ctx=dev())}
    initializer = mx.init.Uniform(0.07)
    exe, params, _ = \
        SGLD(sym=net, data_inputs=data_inputs,
             X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=50000,
             initializer=initializer,
             learning_rate=1E-4,
             #         lr_scheduler=mx.lr_scheduler.FactorScheduler(100000, 0.5),
             prior_precision=0.1,
             burn_in_iter_num=1000,
             thin_interval=10,
             task='regression',
             minibatch_size=minibatch_size, dev=dev())


def run_toy_DistilledSGLD():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = 1
    teacher_noise_precision = 1.0
    teacher_net = get_toy_sym(True, teacher_noise_precision)
    student_net = get_toy_sym(False)
    data_shape = (minibatch_size,) + X.shape[1::]
    teacher_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                           'teacher_output_label': nd.zeros((minibatch_size, 1), ctx=dev())}
    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev())}
    #                   'softmax_label': nd.zeros((minibatch_size, 10), ctx=dev())}
    teacher_initializer = mx.init.Uniform(0.07)
    student_initializer = mx.init.Uniform(0.07)
    student_grad_f = lambda student_outputs, teacher_pred: \
        regression_student_grad(student_outputs, teacher_pred, teacher_noise_precision)
    student_exe, student_params, _ = \
        DistilledSGLD(teacher_sym=teacher_net, student_sym=student_net,
                      teacher_data_inputs=teacher_data_inputs,
                      student_data_inputs=student_data_inputs,
                      X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=80000,
                      teacher_initializer=teacher_initializer,
                      student_initializer=student_initializer,
                      teacher_learning_rate=1E-4, student_learning_rate=0.01,
                      # teacher_lr_scheduler=mx.lr_scheduler.FactorScheduler(100000, 0.5),
                      student_lr_scheduler=mx.lr_scheduler.FactorScheduler(8000, 0.8),
                      student_grad_f=student_grad_f,
                      teacher_prior_precision=0.1, student_prior_precision=0.001,
                      perturb_deviation=0.1, minibatch_size=minibatch_size, task='regression',
                      dev=dev())


def run_toy_HMC():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = Y.shape[0]
    noise_precision = 1 / 9.0
    net = get_toy_sym(True, noise_precision)
    data_shape = (minibatch_size,) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'teacher_output_label': nd.zeros((minibatch_size, 1), ctx=dev())}
    initializer = mx.init.Uniform(0.07)
    sample_pool = HMC(net, data_inputs=data_inputs, X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                      sample_num=300000, initializer=initializer, prior_precision=1.0,
                      learning_rate=1E-3, L=10, dev=dev())


def run_synthetic_SGLD():
    theta1 = 0
    theta2 = 1
    sigma1 = numpy.sqrt(10)
    sigma2 = 1
    sigmax = numpy.sqrt(2)
    X = load_synthetic(theta1=theta1, theta2=theta2, sigmax=sigmax, num=100)
    minibatch_size = 1
    total_iter_num = 1000000
    lr_scheduler = SGLDScheduler(begin_rate=0.01, end_rate=0.0001, total_iter_num=total_iter_num,
                                 factor=0.55)
    optimizer = mx.optimizer.create('sgld',
                                    learning_rate=None,
                                    rescale_grad=1.0,
                                    lr_scheduler=lr_scheduler,
                                    wd=0)
    updater = mx.optimizer.get_updater(optimizer)
    theta = mx.random.normal(0, 1, (2,), mx.cpu())
    grad = nd.empty((2,), mx.cpu())
    samples = numpy.zeros((2, total_iter_num))
    start = time.time()
    for i in xrange(total_iter_num):
        if (i + 1) % 100000 == 0:
            end = time.time()
            print("Iter:%d, Time spent: %f" % (i + 1, end - start))
            start = time.time()
        ind = numpy.random.randint(0, X.shape[0])
        synthetic_grad(X[ind], theta, sigma1, sigma2, sigmax, rescale_grad=
        X.shape[0] / float(minibatch_size), grad=grad)
        updater('theta', grad, theta)
        samples[:, i] = theta.asnumpy()
    plt.hist2d(samples[0, :], samples[1, :], (200, 200), cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    numpy.random.seed(100)
    mx.random.seed(100)
    parser = argparse.ArgumentParser(
        description="Examples in the paper [NIPS2015]Bayesian Dark Knowledge and "
                    "[ICML2011]Bayesian Learning via Stochastic Gradient Langevin Dynamics")
    parser.add_argument("-d", "--dataset", type=int, default=1,
                        help="Dataset to use. 0 --> TOY, 1 --> MNIST, 2 --> Synthetic Data in "
                             "the SGLD paper")
    parser.add_argument("-l", "--algorithm", type=int, default=2,
                        help="Type of algorithm to use. 0 --> SGD, 1 --> SGLD, other-->DistilledSGLD")
    parser.add_argument("-t", "--training", type=int, default=50000,
                        help="Number of training samples")
    args = parser.parse_args()
    training_num = args.training
    if args.dataset == 1:
        if 0 == args.algorithm:
            run_mnist_SGD(training_num)
        elif 1 == args.algorithm:
            run_mnist_SGLD(training_num)
        else:
            run_mnist_DistilledSGLD(training_num)
    elif args.dataset == 0:
        if 1 == args.algorithm:
            run_toy_SGLD()
        elif 2 == args.algorithm:
            run_toy_DistilledSGLD()
        elif 3 == args.algorithm:
            run_toy_HMC()
    else:
        run_synthetic_SGLD()
