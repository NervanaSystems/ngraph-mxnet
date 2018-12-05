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

# Environment to build and run unit tests for ngraph-mxnet on gpu
# with cuda 8
# with libcudnn7-dev_7.0.1.13
# with capabilities for python 2.7 and python 3
# Author: Lam Nguyen

#FROM nvidia/cuda:8.0-devel-ubuntu16.04
FROM nvidia/cuda:8.0-cudnn7-devel

COPY install/cpp.sh install/
RUN install/cpp.sh


RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# try to get around issue with badsig
RUN rm /etc/apt/sources.list.d/cuda.list

# removing nvidia-ml.list file to avoid apt-get update error
# "The method driver /usr/lib/apt/methods/https could not be found."
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

#RUN apt-get update && apt-get install -y --no-install-recommends \
#            libcudnn7-dev_7.0.1.13-1+cuda8.0_amd64.deb && \
#    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y curl apt-transport-https ca-certificates && \
    apt-get clean autoclean && \
    apt-get autoremove -y
RUN curl http://developer.download.nvidia.com/compute/cuda/repos/GPGKEY | apt-key add -

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/include:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

#Install MxNet Dependencies:
RUN apt-get update && apt-get install -y build-essential git \
        bzip2 wget coreutils libjasper1 libjpeg8 libpng12-0 \
        libsox-dev libgtest-dev cpio pandoc curl libtcmalloc-minimal4 \
        libssl-dev libffi-dev \
        libopencv-dev curl gcc libatlas-base-dev \
        python python-pip python-dev \
        python3 python3-pip python3-dev \
        python-opencv graphviz python-scipy \
        python-sklearn libopenblas-dev clang-3.9 \
        pciutils \
        clang-format-3.9 virtualenv cmake \
        sudo && \
        apt-get clean autoclean && \
        apt-get autoremove -y

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
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
