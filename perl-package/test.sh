#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

MXNET_HOME=${PWD}
export PERL5LIB=${MXNET_HOME}/perl5/lib/perl5

cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
perl Makefile.PL INSTALL_BASE=${MXNET_HOME}/perl5
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
perl Makefile.PL INSTALL_BASE=${MXNET_HOME}/perl5
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-MXNet/
perl Makefile.PL INSTALL_BASE=${MXNET_HOME}/perl5
make test TEST_VERBOSE=1 || exit -1 # Add debug output to test log
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-MXNet-Gluon-Contrib/
perl Makefile.PL INSTALL_BASE=${MXNET_HOME}/perl5
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-MXNet-Gluon-ModelZoo/
perl Makefile.PL INSTALL_BASE=${MXNET_HOME}/perl5
make test TEST_VERBOSE=1 || exit -1

