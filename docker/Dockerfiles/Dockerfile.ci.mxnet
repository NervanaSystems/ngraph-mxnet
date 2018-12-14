
# *******************************************************************************
# * Copyright 2018 Intel Corporation
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# ********************************************************************************

# Environment to build and run unit tests for ngraph-mxnet on ubuntu1604
# with gcc 5.4 by defaul
# with clang 3.9
# with capabilities for python 2.7 and python 3
# Author:  Kim Janik / Lam Nguyen


FROM ubuntu:16.04

ARG gccversion

#Install MxNet Dependencies:
RUN apt-get update && apt-get install -y build-essential git \
        curl \
        libatlas-base-dev \
        python \
        python-pip \
        python-dev \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \      
        python-opencv \
        graphviz \
        python-scipy \
        python-sklearn \
        libopenblas-dev \
        clang-3.9 \
        clang-format-3.9 \
        virtualenv \
        cmake \
        sudo 

RUN if [ "$gccversion" = "gcc-5" ]; then \
        apt-get update && apt-get install -y g++-5 gcc-5 \
        libopencv-dev ; \
else \
       apt-get update && apt-get install -y g++-4.8 gcc-4.8 wget unzip zip ; \
       #apt-get update && apt-get install -y g++-4.8 gcc-4.8 wget unzip zip && \
       #libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
       #wget https://codeload.github.com/opencv/opencv/zip/2.4.13 && \
       #unzip 2.4.13 && \
       #cd opencv-2.4.13 && \
       #mkdir release && \
       #cd release && \
       #cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D #CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_CXX_COMPILER=/usr/bin/g++-4.8 -D CMAKE_C_COMPILER=/usr/bin/gcc-4.8 .. && \
       #make -j $(nproc) && \
       #make install ; \
fi

RUN cmake --version
RUN make --version
RUN gcc --version
RUN c++ --version

RUN pip install --upgrade pip
RUN pip install numpy

# We include psutil
RUN pip install psutil

# We include pytest
RUN pip install --upgrade pytest

# We include pytest-xdist to speed up the testing
RUN pip install pytest-xdist

# Copy in the run-as-user.sh script
# This will allow the builds, which are done in a mounted directory, to
# be run as the user who runs "docker run".  This then allows the mounted
# directory to be properly deleted by the user later (e.g. by jenkins).
WORKDIR /home
ADD scripts/run-as-user.sh /home/run-as-user.sh
