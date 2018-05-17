# Tutorials

MXNet has two primary high-level interfaces for its deep learning engine: the Gluon API and the Module API. Tutorials for each are provided below.

`TL;DR:` If you are new to deep learning or MXNet, you should start with the Gluon tutorials.

The difference between the two is an imperative versus symbolic programming style. Gluon makes it easy to prototype, build, and train deep learning models without sacrificing training speed by enabling both (1) intuitive imperative Python code development and (2) faster execution by automatically generating a symbolic execution graph using the hybridization feature.

The Gluon and Module tutorials are in Python, but you can also find a variety of other MXNet tutorials, such as R, Scala, and C++ in the [Other Languages API Tutorials](#other-mxnet-api-tutorials) section below.

[Example scripts and applications](#example-scripts-and-applications) as well as [contribution](#contributing-tutorials) info is below.

<script type="text/javascript" src='../_static/js/options.js'></script>


## Python API Tutorials

<!-- Gluon vs Module -->
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active" style="font-size:22px">Gluon</button>
  <button type="button" class="btn btn-default opt"   style="font-size:22px">Module</button>
</div>


<!-- Levels -->
<div class="gluon module">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Introduction</button>
  <button type="button" class="btn btn-default opt">Applications</button>
</div>
</div>


<!-- introduction Topics -->
<div class="introduction">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Basics</button>
  <button type="button" class="btn btn-default opt">Neural Networks</button>
  <button type="button" class="btn btn-default opt">Advanced</button>
</div>
</div>


<!-- Intermediate Topics
<div class="intermediate">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Image Recognition</button>
  <button type="button" class="btn btn-default opt">Human Language</button>
  <button type="button" class="btn btn-default opt">Recommender Systems</button>
  <button type="button" class="btn btn-default opt">Customization</button>
</div>
</div>
-->

<!-- Advanced Topics
<div class="advanced">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Distributed Training</button>
  <button type="button" class="btn btn-default opt">Optimization</button>
  <button type="button" class="btn btn-default opt">Adversarial Networks</button>
</div>
</div>
-->
<!-- END - Main Menu -->
<hr>

<div class="gluon">
<div class="introduction">


<div class="basics">

- [Manipulate data the MXNet way with ndarray](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html)

- [Automatic differentiation with autograd](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)

- [Linear regression with gluon](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html)

- [Serialization - saving, loading and checkpointing](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html)

- [Gluon Datasets and DataLoaders](http://mxnet.incubator.apache.org/tutorials/gluon/datasets.html)

</div>


<div class="neural-networks">

- [Multilayer perceptrons in gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html)

- [Multi-class object detection using CNNs in gluon](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)

- [Advanced RNNs with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html)

</div>


<div class="advanced">

- [Plumbing: A look under the hood of gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html)

- [Designing a custom layer with gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html)

- [Block and Parameter naming](/tutorials/gluon/naming.html)

- [Fast, portable neural networks with Gluon HybridBlocks](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)

- [Training on multiple GPUs with gluon](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html)

- [Applying data augmentation](/tutorials/gluon/data_augmentation.html)

</div>

</div> <!--end of introduction-->


<div class="applications">

- [Creating custom operators with numpy](/tutorials/gluon/customop.html)

- [Handwritten digit recognition (MNIST)](/tutorials/gluon/mnist.html)

- [Hybrid network example](/tutorials/gluon/hybrid.html)

- [Neural network building blocks with gluon](/tutorials/gluon/gluon.html)

- [Simple autograd example](/tutorials/gluon/autograd.html)

- [Data Augmentation with Masks (for Object Segmentation)](/tutorials/python/data_augmentation_with_masks.html)

- [Inference using an ONNX model](/tutorials/onnx/inference_on_onnx_model.html)

- [Fine-tuning an ONNX model on Gluon](/tutorials/onnx/fine_tuning_gluon.html)

</div> <!--end of applications-->

</div> <!--end of gluon-->


<div class="module">


<div class="introduction">


<div class="basics">

- [Imperative tensor operations on CPU/GPU](/tutorials/basic/ndarray.html)

- [NDArray Indexing](/tutorials/basic/ndarray_indexing.html)

- [Symbol API](/tutorials/basic/symbol.html)

- [Module API](/tutorials/basic/module.html)

- [Iterators - Loading data](/tutorials/basic/data.html)

</div>


<div class="neural-networks">

- [Linear regression](/tutorials/python/linear-regression.html)

- [MNIST - handwriting recognition](/tutorials/python/mnist.html)

- [Large scale image classification](/tutorials/vision/large_scale_classification.html)

<!-- broken #9532
- [Image recognition](/tutorials/python/predict_image.html)
-->
</div>


<div class="advanced">

- [NDArray in Compressed Sparse Row storage format](/tutorials/sparse/csr.html)

- [Sparse gradient updates](/tutorials/sparse/row_sparse.html)

- [Train a linear regression model with sparse symbols](/tutorials/sparse/train.html)

- [Applying data augmentation](/tutorials/python/data_augmentation.html)

- [Types of data augmentation](/tutorials/python/types_of_data_augmentation.html)

</div>

</div> <!--end of introduction-->


<div class="applications">

- [Connectionist Temporal Classification](../tutorials/speech_recognition/ctc.html)

- [Distributed key-value store](/tutorials/python/kvstore.html)

- [Fine-tuning a pre-trained ImageNet model with a new dataset](/faq/finetune.html)

- [Generative Adversarial Networks](/tutorials/unsupervised_learning/gan.html)

- [Matrix factorization in recommender systems](/tutorials/python/matrix_factorization.html)

- [Text classification (NLP) on Movie Reviews](/tutorials/nlp/cnn.html)

- [Importing an ONNX model into MXNet](http://mxnet.incubator.apache.org/tutorials/onnx/super_resolution.html) 

</div> <!--end of applications-->

</div> <!--end of module-->


<hr>

## Other Languages API Tutorials


<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">C</button>
  <button type="button" class="btn btn-default opt">Scala</button>
  <button type="button" class="btn btn-default opt">R</button>
</div>
<hr>

<div class="c">

- [MNIST with the MXNet C++ API](/tutorials/c%2B%2B/basics.html)
</div> <!--end of c++-->


<div class="r">

- [NDArray: Vectorized Tensor Computations on CPUs and GPUs with R](/tutorials/r/ndarray.html)
- [Symbol API with R](/tutorials/r/symbol.html)
- [Custom Iterator](/tutorials/r/CustomIterator.html)
- [Callback Function](/tutorials/r/CallbackFunction.html)
- [Five minute neural network](/tutorials/r/fiveMinutesNeuralNetwork.html)
- [MNIST with R](/tutorials/r/mnistCompetition.html)
- [Classify images via R with a pre-trained model](/tutorials/r/classifyRealImageWithPretrainedModel.html)
- [Char RNN Example with R](/tutorials/r/charRnnModel.html)
- [Custom loss functions in R](/tutorials/r/CustomLossFunction.html)


</div> <!--end of r-->


<div class="scala">

- [Setup your MXNet with Scala on IntelliJ](/tutorials/scala/mxnet_scala_on_intellij.html)
- [MNIST with the Scala API](/tutorials/scala/mnist.html)
- [Use Scala to build a Long Short-Term Memory network that generates Barack Obama's speech patterns](/tutorials/scala/char_lstm.html)

</div> <!--end of scala-->

<hr>


## Example Scripts and Applications

More tutorials and examples are available in the [GitHub repository](https://github.com/apache/incubator-mxnet/tree/master/example).


## Learn More About Gluon!

Most of the Gluon tutorials are hosted on [gluon.mxnet.io](http://gluon.mxnet.io), and you may want to follow the chapters on directly the Gluon site.


## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, [review these details](https://github.com/apache/incubator-mxnet/tree/master/example#contributing) on example and tutorial writing.
