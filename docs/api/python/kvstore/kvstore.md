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

# KVStore API

.. note:: Direct interactions with ``KVStore`` are dangerous and not recommended.

## Basic Push and Pull

Provides basic operation over multiple devices (GPUs) on a single device.

### Initialization

Let's consider a simple example. It initializes
a (`int`, `NDArray`) pair into the store, and then pulls the value out.

```python
>>> kv = mx.kv.create('local') # create a local kv store.
>>> shape = (2,3)
>>> kv.init(3, mx.nd.ones(shape)*2)
>>> a = mx.nd.zeros(shape)
>>> kv.pull(3, out = a)
>>> print a.asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

### Push, Aggregation, and Updater

For any key that's been initialized, you can push a new value with the same shape to the key, as follows:

```python
>>> kv.push(3, mx.nd.ones(shape)*8)
>>> kv.pull(3, out = a) # pull out the value
>>> print a.asnumpy()
[[ 8.  8.  8.]
 [ 8.  8.  8.]]
```

The data that you want to push can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore first sums all of these
values, and then pushes the aggregated value, as follows:

```python
>>> gpus = [mx.gpu(i) for i in range(4)]
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.push(3, b)
>>> kv.pull(3, out = a)
>>> print a.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
```

For each push command, KVStore applies the pushed value to the value stored by an
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged.

```python
>>> def update(key, input, stored):
>>>     print "update on key: %d" % key
>>>     stored += input * 2
>>> kv._set_updater(update)
>>> kv.pull(3, out=a)
>>> print a.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
>>> kv.push(3, mx.nd.ones(shape))
update on key: 3
>>> kv.pull(3, out=a)
>>> print a.asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

### Pull

You've already seen how to pull a single key-value pair. Similar to the way that you use the push command, you can
pull the value into several devices with a single call.

```python
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.pull(3, out = b)
>>> print b[1].asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

## List Key-Value Pairs

All of the operations that we've discussed so far are performed on a single key. KVStore also provides
the interface for generating a list of key-value pairs. For a single device, use the following:

```python
>>> keys = [5, 7, 9]
>>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
>>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
update on key: 5
update on key: 7
update on key: 9
>>> b = [mx.nd.zeros(shape)]*len(keys)
>>> kv.pull(keys, out = b)
>>> print b[1].asnumpy()
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
```

For multiple devices:

```python
>>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
>>> kv.push(keys, b)
update on key: 5
update on key: 7
update on key: 9
>>> kv.pull(keys, out = b)
>>> print b[1][1].asnumpy()
[[ 11.  11.  11.]
 [ 11.  11.  11.]]
```



## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.kvstore
    :members:
```

<script>auto_index("api-reference");</script>
