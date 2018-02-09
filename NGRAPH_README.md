# nGraph - MXNet Integration
## Compilation instructions for Ubuntu 16.04

### Install Ubuntu Prerequisites
sudo apt-get update
sudo apt-get install -y build-essential git libopencv-dev curl gcc libatlas-base-dev python python-pip python-dev python-opencv graphviz python-scipy python-sklearn libopenblas-dev

### Build google test and install it into /usr/local/lib

git clone https://github.com/google/googletest.git

cd googletest/ && cmake . && make -j$(nproc) && sudo make install

### Clone ngraph-cpp
Compile and install with cmake according to the readme.

This will install ngraph-cpp to $HOME/ngraph_dist

Set the path to the ngraph libraries

export LD_LIBRARY_PATH=$HOME/ngraph_dist/lib/

### Clone the ngraph-mxnet repository recursively and checkout the ngraph-integration-dev branch
Edit make/config.mk file to use_ngraph and point to your ngraph installation

USE_NGRAPH = 1

NGRAPH_DIR = $(HOME)/ngraph_dist

Ensure that the config file has nnpack and mklml disabled (to isolate testing)

Compile mxnet with make -j $(nproc)

### Set up a virtual environment

Python 2: virtualenv -p python2.7 .venv && . .venv/bin/activate

Python 3: python3 -m venv .venv && . .venv/bin/activate

### Install the python package

cd python && pip install -e . && cd ../

### Run mnist

python example/image-classification/train_mnist.py

## Release Notes:
This release enables examples/image_classification/train_mnist.py and examples/rnn/lstm_bucketing.py on CPU. Other models are under development but not fully supported at this time.

## Test status
Integration testing to date (2/8/2018) has focused on "tests/cpp/*" and "tests/python/unittest/*". Of these tests, we see the following failures.

### Ngraph changes the number of nodes in the graph, so the assumptions in this test are no longer valid.
tests/python/unittest/test_module.py::test_monitor

### Profiler integration is ongoing but incomplete, so profiler fails
tests/python/unittest/test_profiler.py::test_profiler

### The current integration only returns dense arrays, so these tests fail when checking the ouput for sparse tensors.
tests/python/unittest/test_sparse_operator.py::test_elemwise_binary_ops
tests/python/unittest/test_sparse_operator.py::test_sparse_mathematical_core
tests/python/unittest/test_sparse_operator.py::test_sparse_unary_with_numerics

### We haven't yet integrated ngraph into the debug string, so memory allocation isn't properly supported and test_zero_prop fails
tests/python/unittest/test_symbol.py::test_zero_prop

Integration testing on other python tests is forthcoming
