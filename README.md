# nGraph - MXNet Integration

## Installation

Warning, this package has only been tested on Ubuntu 16.04 with Intel CPUs from the Haswell line and later. Use on other platforms/CPUs may or may not work.

1. **Install Ubuntu prerequisites**
   Run the following commands from a command-line:

     ``` sh
      sudo apt-get update
      sudo apt-get install -y \
       libatlas-base-dev \
       libopenblas-dev \
       libopencv-dev\
       python \
       python-pip \
       python3 \
       python3-pip \
       virtualenv
     ```
1. **(Optional) Activate a Python virtual environment.**

   Activate the Python virtual environment into which you wish
   to process with ngraph-mxnet bridge installation.

   (Please visit http://python.org for more information on Python virtual
   environments.)

   For example:
   ``` sh
    source ~/path/to/my/venv/bin/activate
   ```

1.  **Install the pip package.**
   
  
   ``` sh
    pip install ngraph-mxnet
   ```

## Building with nGraph support
MXnet's experimental support for the Intel nGraph graph compiler can be enabled
using MXnet's Makefile-based build system.

When building MXnet with experimental nGraph integration enabled, MXnet's build
system builds its own copy of the nGraph-supplied libraries.  Upon successful
completion of an nGraph-enabled build, these libraries and related symbolic links
can be found in the same build directory as `libmxnet.so`.

The experimental MXnet nGraph integration does not currently support externally
supplied builds of nGraph's libraries to be used by MXnet.  However, it is possible
to use control which particular version of the nGraph source code is built by
executing an appropriate `git checkout ...` command in the `3rdparty/ngraph-mxnet-bridge`
MXnet source tree.

Some of the Make-variable influential for experimental nGraph MXnet integration
are as follows:
- `USE_NGRAPH` - If set to `1`, build MXnet with experimental nGraph integration
  enabled.
- `NGRAPH_EXTRA_CMAKE_FLAGS` - nGraph uses a CMake-based build system. This variable
  can provide additional command-line arguments to the `cmake` invocation used to
  configure nGraph for compilation.
- `NGRAPH_EXTRA_MAKE_FLAGS` - This variable can provide additional command-line
  arguments to the invocation of `make` used to compile and link nGraph.

Please see the files `3rdparty/ngraph-mxnet-bridge/ngraph.mk` and `make/config.mk` for more details.


## Compilation instructions for Ubuntu 16.04

If you have not already done so, create a clone on the local file system of
the official nGraph-enabled Apache MXNet repository:

``` sh
  git clone --recursive https://github.com/NervanaSystems/ngraph-mxnet.git
```

In the instructions below, the root directory of the cloned repository shall
be referred to as `MXNET_ROOT`.

1. **Install Ubuntu prerequisites**
   Run the following commands from a command-line:

     ``` sh
     sudo apt-get update
     sudo apt-get install -y \
       build-essential \
       git \
       graphviz \
       libatlas-base-dev \
       libopenblas-dev \
       libopencv-dev\
       python \
       python-dev \
       python-opencv \
       python-pip \
       python-scipy \
       python-sklearn \
       python3-pip \
       virtualenv
     ```
1. **Build the nGraph-MXNet libraries**

   The build system for nGraph-enabled MXNet uses GNU Makefiles and is performed
   in the MXnet source directory.

   The simplest invocation of the build system is as follows:

     ``` sh
     cd MXNET_ROOT
     make
     ```

   Successful completion of the build process results in the creation of the
   file the files `MXNET/lib/libmxnet.a` and `MXNET/lib/libmxnet.so`.

   The build process is influenced by Make variables whose values can be set in
   several ways:
   - customization of the file `make/config.mk`
   - definition of environment variables with the same name
   - specification on `make` command line.

   Please see the GNU Make program documentation for more information.

   Here's an example of a typical build command:

   ``` sh
   cd MXNET_ROOT
   make USE_NGRAPH=1 USE_CUDA=0 DEBUG=0 -j
     ```
1. **(Optional) Install the MXNet Python bindings**

   Once `libmxnet.so` has been built, one can optionally install Python bindings
   for MXNet into a Python environment as follows.
   1. *(Optional) Activate a Python virtual environment.*

      MXNet's Python bindings are compatible with Python virtual environments.
      (Please visit http://python.org for more information on Python virtual
      environments.)

      If desired, activate the Python virtual environment into which you wish
      to install MXNet's Python bindings.

      For example:
         ``` sh
         source ~/path/to/my/venv/bin/activate
         ```
   1. *Execute the following commands*

        ``` sh
        cd MXNET_ROOT/python
        pip install -e .
        ```
   1. *(Optional)  Verify functionality of installed Python bindings*
        1. If the `pip install -e .` command (see above) installed the
           Python bindings into a Python virtual environment, ensure that
           Python virtual environment is currently activated.

           Conversely, if the Python bindings were installed with no
           virtual environment activated, ensure that no virtual environment
           is active in the current shell.
        1. Run the MNIST example script
             ``` sh
             cd MXNET_ROOT
             python example/image-classification/train_mnist.py
             ```

## Distributed training
MPI is required for multi-CPU support. Download Open MPI from [here](https://www.open-mpi.org/).

Use `USE_NGRAPH_DISTRIBUTED=1` and `NGRAPH_EXTRA_CMAKE_FLAGS=-DNGRAPH_DISTRIBUTED_ENABLE=1`
in order for MXNet to build with nGraph distributed support.

Here's an example of a typical build command for distributed training:

``` sh
cd MXNET_ROOT
make USE_NGRAPH=1 USE_NGRAPH_DISTRIBUTED=1 NGRAPH_EXTRA_CMAKE_FLAGS=-DNGRAPH_DISTRIBUTED_ENABLE=1 USE_CUDA=0 DEBUG=0 -j
```

Here's an example to run ResNet-50 on two nodes:

``` sh
export MXNET_NGRAPH_GLUON=1
export MXNET_ENGINE_TYPE=NaiveEngine
mpirun -map-by node -x MXNET_NGRAPH_GLUON -x MXNET_ENGINE_TYPE -hostfile hosts -np 2 python MXNET_ROOT/example/image-classification/train_cifar10.py --network resnet --num-layers 50 --kv-store ngraph
```

## Runtime environment variables
Some environment variables influence the behavior of the
nGraph-enabled MXNet software and supporting libraries.  Here is a partial list of those variables:

| Variable  | Description |
| :-------- | :---------- |
| `OMP_NUM_THREADS`            | Suggested value: `16`.  For more information please see [here](https://software.intel.com/en-us/mkl-windows-developer-guide-setting-the-number-of-threads-using-an-openmp-environment-variable) |
| `KMP_AFFINITY`               | Suggested value: `granularity=fine,compact,1,0`.  For more information please see [here](https://software.intel.com/en-us/node/522691). |
| `MXNET_NGRAPH_VERBOSE_GRAPH` | When set to `1`, nGraph-enabled MXNet will create in the current directory a JSON file representing each subgraph being compiled by the nGraph library.  Each of these JSON files is a graph serialization that can be loaded by nGraph's `ngraph::deserialize`  functions. |

## Release Notes

### Supported models
The following models are known to run successfully in this release:
* `example/rnn/bucketing/lstm_bucketing.py`
* `example/image-classification/train_mnist.py`
* `example/image-classification/train_cifar10.py`

Other models are under development are not guaranteed to run to completion or converge.
This is a temporary limitation expected to be lifted in a future release.

### Supported nGraph back-ends
The nGraph library supports a number of backends, including `"CPU"`, `"INTERPETER"`, and `"GPU"`.
The supported models listed above explicitly use nGraph's `"CPU"` backend, and may not function
properly if altered to use different nGraph back-ends.
This is a temporary limitation expected to be lifted in a future release.

### Test status
Integration testing to date (11/15/2018) has focused on `tests/cpp/*` and `tests/python/unittest/*`.
Of these tests, we see the following failures.

#### This test fails with relative errors of <1e-4 on a limit of 1e-5.
- `tests/python/unittest/test_gluon.py::test_export`

#### These test fail with python errors
- `tests/python/unittest/test_image.py::test_det_augmenters`
- `tests/python/unittest/test_image.py::test_image_detiter`
- `tests/python/unittest/test_image.py:test_imageiter`

#### nGraph changes the number of nodes in the graph, so the assumptions in this test are no longer valid.
- `tests/python/unittest/test_module.py::test_monitor`

Integration testing on other python tests are forthcoming.

