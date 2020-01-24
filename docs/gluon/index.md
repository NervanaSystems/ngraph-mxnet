<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# About Gluon

![gluon logo](https://github.com/dmlc/web-data/blob/master/mxnet/image/image-gluon-logo.png?raw=true)

Based on the [the Gluon API specification](https://github.com/gluon-api/gluon-api), the new Gluon library in Apache MXNet provides a clear, concise, and simple API for deep learning. It makes it easy to prototype, build, and train deep learning models without sacrificing training speed. [Install the latest version of MXNet](/install/) to get access to Gluon.

To get started with Gluon, checkout the following resources and tutorials:
* [60-minute Gluon Crash Course](https://gluon-crash-course.mxnet.io/) - six 10-minute lessons on using Gluon
* [GluonCV Toolkit](https://gluon-cv.mxnet.io/) - implementations of state of the art deep learning algorithms in **Computer Vision (CV)**
* [GluonNLP Toolkit](https://gluon-nlp.mxnet.io/) - implementations of state of the art deep learning algorithms in **Natural Language Processing (NLP)**
* [Gluon: The Straight Dope](https://gluon.mxnet.io/) - notebooks designed to teach deep learning from the ground up, all using the Gluon API

<br/>
<div class="boxed">
    Advantages
</div>

1. Simple, Easy-to-Understand Code: Gluon offers a full set of plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers.

2. Flexible, Imperative Structure: Gluon does not require the neural network model to be   rigidly defined, but rather brings the training algorithm and model closer together to provide flexibility in the development process.

3. Dynamic Graphs: Gluon enables developers to define neural network models that are dynamic, meaning they can be built on the fly, with any structure, and using any of Python’s native control flow.

4. High Performance: Gluon provides all of the above benefits without impacting the training speed that the underlying engine provides.

<br/>
<div class="boxed">
    The Straight Dope
</div>

The community is also working on parallel effort to create a foundational resource for learning about machine learning. The Straight Dope is a book composed of introductory as well as advanced tutorials – all based on the Gluon interface. For example,

* [Learn about machine learning basics](http://gluon.mxnet.io/chapter01_crashcourse/introduction.html).
* [Develop and train a simple neural network model](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html).
* [Implement a Recurrent Neural Network (RNN) model for Language Modeling](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/simple-rnn.html).

<br/>
<div class="boxed">
    Code Examples
</div>

**__Simple, Easy-to-Understand Code__**

Use plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers:

```
net = gluon.nn.Sequential()
# When instantiated, Sequential stores a chain of neural network layers.
# Once presented with data, Sequential executes each layer in turn, using
# the output of one layer as the input for the next
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
    net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net.add(gluon.nn.Dense(num_outputs))
```
<br/>

**__Flexible, Imperative Structure__**

Prototype, build, and train neural networks in fully imperative manner using the MXNet autograd package and the Gluon trainer method:

```
epochs = 10

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        with autograd.record():
            output = net(data) # the forward iteration
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
```

<br/>
**__Dynamic Graphs__**

Build neural networks on the fly for use cases where neural networks must change in size and shape during model training:

```
def forward(self, F, inputs, tree):
    children_outputs = [self.forward(F, inputs, child)
                        for child in tree.children]
    #Recursively builds the neural network based on each input sentence’s
    #syntactic structure during the model definition and training process
```
<br/>
**__High Performance__**

Easily cache the neural network to achieve high performance by defining your neural network with ``HybridSequential`` and calling the ``hybridize`` method:

```
net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(2))
```

```
net.hybridize()
```

## Learn More

* [Gluon API Documentation](/api/python/gluon/gluon.html)
* [Gluon Tutorials](/tutorials/)
* [60-minute Gluon Crash Course](https://gluon-crash-course.mxnet.io/)
* [GluonCV Toolkit](https://gluon-cv.mxnet.io/)
* [GluonNLP Toolkit](https://gluon-nlp.mxnet.io/)
* [Gluon: The Straight Dope](https://gluon.mxnet.io/)
