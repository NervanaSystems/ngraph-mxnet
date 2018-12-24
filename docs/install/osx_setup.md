# Installing MXNet from source on OS X (Mac)

**NOTE:** For prebuild MXNet with Python installation, please refer to the [new install guide](http://mxnet.io/install/index.html).

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file and submit a build request with the ```make``` command.

## Prepare Environment for GPU Installation

This section is optional. Skip to next section if you don't plan to use GPUs. If you plan to build with GPU, you need to set up the environment for CUDA and cuDNN.

First, download and install [CUDA 8 toolkit](https://developer.nvidia.com/cuda-toolkit).

Once you have the CUDA Toolkit installed you will need to set up the required environment variables by adding the following to your ~/.bash_profile file:

```bash
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$DYLD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
```

Reload ~/.bash_profile file and install dependencies:
```bash
    . ~/.bash_profile
    brew install coreutils
    brew tap caskroom/cask
```

Then download [cuDNN 5](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cudnn root directory. Move the header files and libraries to your local CUDA Toolkit folder:

```bash
    $ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
    $ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
    $ sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/
```

Now we can start to build MXNet.

## Build the Shared Library

### Install MXNet dependencies
Install the dependencies, required for MXNet, with the following commands:
- [Homebrew](http://brew.sh/)
- OpenBLAS and homebrew/core (for linear algebraic operations)
- OpenCV (for computer vision operations)

```bash
	# Paste this command in Mac terminal to install Homebrew
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

	# Insert the Homebrew directory at the top of your PATH environment variable
	export PATH=/usr/local/bin:/usr/local/sbin:$PATH
```

```bash
	brew update
	brew install pkg-config
	brew install graphviz
	brew install openblas
	brew tap homebrew/core
	brew install opencv

	# If building with MKLDNN
	brew install llvm

	# Get pip
	easy_install pip
	# For visualization of network graphs
	pip install graphviz
	# Jupyter notebook
	pip install jupyter
```

### Build MXNet Shared Library
After you have installed the dependencies, pull the MXNet source code from Git and build MXNet to produce an MXNet library called ```libmxnet.so```. You can clone the repository as described in the following code block, or you may try the <a href="download.html">download links</a> for your desired MXNet version.

The file called ```osx.mk``` has the configuration required for building MXNet on OS X. First copy ```make/osx.mk``` into ```config.mk```, which is used by the ```make``` command:

```bash
    git clone --recursive https://github.com/apache/incubator-mxnet ~/mxnet
    cd ~/mxnet
    cp make/osx.mk ./config.mk
    echo "USE_BLAS = openblas" >> ./config.mk
    echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
    make -j$(sysctl -n hw.ncpu)
```

To build with MKLDNN

```bash
echo "CC=$(brew --prefix llvm)/bin/clang++" >> ./config.mk
echo "CXX=$(brew --prefix llvm)/bin/clang++" >> ./config.mk
echo "USE_OPENCV=1" >> ./config.mk
echo "USE_OPENMP=1" >> ./config.mk
echo "USE_MKLDNN=1" >> ./config.mk
echo "USE_BLAS=apple" >> ./config.mk
echo "USE_PROFILER=1" >> ./config.mk
LIBRARY_PATH=$(brew --prefix llvm)/lib/ make -j $(sysctl -n hw.ncpu)
```

If building with ```GPU``` support, add the following configuration to config.mk and build:
```bash
    echo "USE_CUDA = 1" >> ./config.mk
    echo "USE_CUDA_PATH = /usr/local/cuda" >> ./config.mk
    echo "USE_CUDNN = 1" >> ./config.mk
    make -j$(sysctl -n hw.ncpu)
```
**Note:** To change build parameters, edit ```config.mk```.


&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for the programming language of your choice:
- [Python](#install-mxnet-for-python)
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Scala](#install-the-mxnet-package-for-scala)
- [Perl](#install-the-mxnet-package-for-perl)

## Install MXNet for Python
To install the MXNet Python binding navigate to the root of the MXNet folder then run the following:

```bash
$ cd python
$ pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

## Install the MXNet Package for R
You have 2 options:
1. Building MXNet with the Prebuilt Binary Package
2. Building MXNet from Source Code

### Building MXNet with the Prebuilt Binary Package
Install OpenCV and OpenBLAS.

```bash
brew install opencv
brew install openblas@0.3.1
```

Add a soft link to the OpenBLAS installation. This example links the 0.3.1 version:

```bash
ln -sf /usr/local/opt/openblas/lib/libopenblasp-r0.3.* /usr/local/opt/openblas/lib/libopenblasp-r0.3.1.dylib
```

Install the latest version (3.5.1+) of R from [CRAN](https://cran.r-project.org/bin/macosx/).
For OS X (Mac) users, MXNet provides a prebuilt binary package for CPUs. The prebuilt package is updated weekly. You can install the package directly in the R console using the following commands:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
```

### Building MXNet from Source Code

Run the following commands to install the MXNet dependencies and build the MXNet R package.

```r
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
```
```bash
    cd R-package
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..
    make rpkg
```

## Install the MXNet Package for Julia
The MXNet package for Julia is hosted in a separate repository, MXNet.jl, which is available on [GitHub](https://github.com/dmlc/MXNet.jl). To use Julia binding it with an existing libmxnet installation, set the ```MXNET_HOME``` environment variable by running the following command:

```bash
	export MXNET_HOME=/<path to>/libmxnet
```

The path to the existing libmxnet installation should be the root directory of libmxnet. In other words, you should be able to find the ```libmxnet.so``` file at ```$MXNET_HOME/lib```. For example, if the root directory of libmxnet is ```~```, you would run the following command:

```bash
	export MXNET_HOME=/~/libmxnet
```

You might want to add this command to your ```~/.bashrc``` file. If you do, you can install the Julia package in the Julia console using the following command:

```julia
	Pkg.add("MXNet")
```

For more details about installing and using MXNet with Julia, see the [MXNet Julia documentation](http://dmlc.ml/MXNet.jl/latest/user-guide/install/).


## Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](./scala_setup.md).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.md) instead.


## Install the MXNet Package for Perl
Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library).
After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    brew install swig
    sudo sh -c 'curl -L https://cpanmin.us | perl - App::cpanminus'
    sudo cpanm -q -n PDL Mouse Function::Parameters Hash::Ordered PDL::CCS

    MXNET_HOME=${PWD}
    export PERL5LIB=${HOME}/perl5/lib/perl5

    cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make
    install_name_tool -change lib/libmxnet.so \
        ${MXNET_HOME}/lib/libmxnet.so \
        blib/arch/auto/AI/MXNetCAPI/MXNetCAPI.bundle
    make install

    cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make
    install_name_tool -change lib/libmxnet.so \
            ${MXNET_HOME}/lib/libmxnet.so \
            blib/arch/auto/AI/NNVMCAPI/NNVMCAPI.bundle
    make install

    cd ${MXNET_HOME}/perl-package/AI-MXNet/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install
```

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/faq/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
