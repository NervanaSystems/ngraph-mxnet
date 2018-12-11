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
# with gcc 4.8.5
# Author:  Lam Nguyen

FROM centos:7

#Install MxNet Dependencies:

RUN yum groupinstall -y "Development Tools" && \
    yum -y --enablerepo=extras install epel-release && \
    yum -y install \
    atlas-devel \
    atlas \
    lapack-devel \
    blas-devel \
    wget \
    glibc-devel.i686 \
    gcc gcc-c++ \
    cmake3 make \
    git \
    sudo \
    which \
    python python-devel python-setuptools \
    python36u python36u-devel python36u-setuptools

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake

RUN wget https://codeload.github.com/opencv/opencv/zip/2.4.13 && \
        unzip 2.4.13 && \
        cd opencv-2.4.13 && \
        mkdir release && \
        cd release && \
        cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
        make -j $(nproc) && \
        make install 

RUN cmake --version
RUN make --version
RUN gcc --version
RUN c++ --version

RUN ls /usr/local/lib/

RUN easy_install pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install virtualenv


# We include psutil
#RUN pip install psutil

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
# Add run-as-user script
ADD scripts/run_as_centos_user.sh           /home/run-as-user.sh
