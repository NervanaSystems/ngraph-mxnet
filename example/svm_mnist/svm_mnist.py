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


#############################################################
## Please read the README.md document for better reference ##
#############################################################
from __future__ import print_function
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Network declaration as symbols. The following pattern was based
# on the article, but feel free to play with the number of nodes
# and with the activation function
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=512)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 512)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)

# Here we add the ultimate layer based on L2-SVM objective
mlp = mx.symbol.SVMOutput(data=fc3, name='svm')

# To use L1-SVM objective, comment the line above and uncomment the line below
# mlp = mx.symbol.SVMOutput(data=fc3, name='svm', use_linear=True)

# Now we fetch MNIST dataset, add some noise, as the article suggests,
# permutate and assign the examples to be used on our network
mnist = fetch_mldata('MNIST original')
mnist_pca = PCA(n_components=70).fit_transform(mnist.data)
noise = np.random.normal(size=mnist_pca.shape)
mnist_pca += noise
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist_pca.shape[0])
X = mnist_pca[p]
Y = mnist.target[p]
X_show = mnist.data[p]

# This is just to normalize the input and separate train set and test set
X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
X_show = X_show[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

# Article's suggestion on batch size
batch_size = 200
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size, label_name='svm_label')
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size, label_name='svm_label')

# Here we instatiate and fit the model for our data
# The article actually suggests using 400 epochs,
# But I reduced to 10, for convinience
mod = mx.mod.Module(
    context = mx.cpu(0),  # Run on CPU 0
    symbol = mlp,         # Use the network we just defined
    label_names = ['svm_label'],
)
mod.fit(
    train_data=train_iter,
    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    batch_end_callback = mx.callback.Speedometer(batch_size, 200),  # Logging module to print out progress
    num_epoch = 10,       # Train for 10 epochs
    optimizer_params = {
        'learning_rate': 0.1,  # Learning rate
        'momentum': 0.9,       # Momentum for SGD with momentum
        'wd': 0.00001,         # Weight decay for regularization
    },
)

# Uncomment to view an example
# plt.imshow((X_show[0].reshape((28,28))*255).astype(np.uint8), cmap='Greys_r')
# plt.show()
# print 'Result:', model.predict(X_test[0:1])[0].argmax()

# Now it prints how good did the network did for this configuration
print('Accuracy:', mod.score(test_iter, mx.metric.Accuracy())[0][1]*100, '%')
