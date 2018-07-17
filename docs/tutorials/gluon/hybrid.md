# Hybrid - Faster training and easy deployment

*Note: a newer version is available [here](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html).*

Deep learning frameworks can be roughly divided into two categories: declarative
and imperative. With declarative frameworks (including Tensorflow, Theano, etc)
users first declare a fixed computation graph and then execute it end-to-end.
The benefit of fixed computation graph is it's portable and runs more
efficiently. However, it's less flexible because any logic must be encoded
into the graph as special operators like `scan`, `while_loop` and `cond`.
It's also hard to debug.

Imperative frameworks (including PyTorch, Chainer, etc) are just the opposite:
they execute commands one-by-one just like old fashioned Matlab and Numpy.
This style is more flexible, easier to debug, but less efficient.

`HybridBlock` seamlessly combines declarative programming and imperative programming
to offer the benefit of both. Users can quickly develop and debug models with
imperative programming and switch to efficient declarative execution by simply
calling: `HybridBlock.hybridize()`.

## HybridBlock

`HybridBlock` is very similar to `Block` but has a few restrictions:

- All children layers of `HybridBlock` must also be `HybridBlock`.
- Only methods that are implemented for both `NDArray` and `Symbol` can be used.
  For example you cannot use `.asnumpy()`, `.shape`, etc.
- Operations cannot change from run to run. For example, you cannot do `if x:`
  if `x` is different for each iteration.

To use hybrid support, we subclass the `HybridBlock`:

```python
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

mx.random.seed(42)

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=2)
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=2)
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            # You can use a Dense layer for fc3 but we do dot product manually
            # here for illustration purposes.
            self.fc3_weight = self.params.get('fc3_weight', shape=(10, 84))

    def hybrid_forward(self, F, x, fc3_weight):
        # Here `F` can be either mx.nd or mx.sym, x is the input data,
        # and fc3_weight is either self.fc3_weight.data() or
        # self.fc3_weight.var() depending on whether x is Symbol or NDArray
        print(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dot(x, fc3_weight, transpose_b=True)
        return x
```

## Hybridize

By default, `HybridBlock` runs just like a standard `Block`. Each time a layer
is called, its `hybrid_forward` will be run:

```python
net = Net()
net.initialize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
```

Hybrid execution can be activated by simply calling `.hybridize()` on the top
level layer. The first forward call after activation will try to build a
computation graph from `hybrid_forward` and cache it. On subsequent forward
calls the cached graph, instead of `hybrid_forward`, will be invoked:

```python
net.hybridize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
```

Note that before hybridize, `print(x)` printed out one NDArray for forward,
but after hybridize, only the first forward printed out a Symbol. On subsequent
forward `hybrid_forward` is not called so nothing was printed.

Hybridize will speed up execution and save memory. If the top level layer is
not a `HybridBlock`, you can still call `.hybridize()` on it and Gluon will try
to hybridize its children layers instead.

`hybridize` also accepts several options for performance tuning. For example, you
can do

```python
net.hybridize(static_alloc=True)
# or
net.hybridize(static_alloc=True, static_shape=True)
```

Please refer to the [API manual](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=hybridize#mxnet.gluon.Block.hybridize)
for details.

## Serializing trained model for deployment

Models implemented as `HybridBlock` can be easily serialized. The serialized
model can be loaded back later or used for deployment
with other language front-ends like C, C++ and Scala. To this end, we simply
use `export` and `SymbolBlock.imports`:

```python
net.export('model', epoch=1)
```

Two files `model-symbol.json` and `model-0001.params` are saved on disk.
You can use other language bindings to load them. You can also load them back
to gluon with `SymbolBlock`:

```python
net2 = gluon.SymbolBlock.imports('model-symbol.json', ['data'], 'model-0001.params')
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
