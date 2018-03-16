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

#!/bin/bash

if [ $# -lt 2 ]; then
	echo "USAGE: prepare_ngraph.sh <NGRAPH_INSTALL_PATH> <NGRAPH_VERSION> [<NGRAPH_CLONE_PATH>]"
	exit 1
fi

# required
NGRAPH_INSTALL_PATH="$1"
NGRAPH_VERSION="$2"

# optional with default 
NGRAPH_CLONE_PATH="$PWD/ngraph"
if [ ! -z "$3" ]; then
	NGRAPH_CLONE_PATH="$3"
fi

# make ngraph clone directory if it doesn't exist
if [ ! -d $NGRAPH_CLONE_PATH ]; then
	mkdir -p $NGRAPH_CLONE_PATH
	if [ $? -ne 0 ]; then
		echo "Could not create nGraph clone directory $NGRAPH_CLONE_PATH"
		exit 1
	fi
fi

# clone ngraph if needed
if [ ! -d $NGRAPH_CLONE_PATH/.git ]; then
	git clone https://github.com/NervanaSystems/ngraph.git $NGRAPH_CLONE_PATH
	if [ $? -ne 0 ]; then
		echo "Could not clone nGraph"
		exit 1
	fi
fi

cd $NGRAPH_CLONE_PATH

# check for nGraph repository
REPO_NAME="$(basename -s .git `git config --get remote.origin.url`)"
if [ "$REPO_NAME" != "ngraph" ]; then
	echo "nGraph clone directory $NGRAPH_CLONE_PATH contains non-ngraph repository $REPO_NAME"
	exit 1
fi

# check if we have to update nGraph
REPO_VERSION="$(git describe)"
if [ "$REPO_VERSION" != *"$NGRAPH_VERSION"* ]; then
	git fetch
	if [ $? -ne 0 ]; then
		echo "Could not fetch nGraph"
		exit 1
	fi

	git checkout $NGRAPH_VERSION
	if [ $? -ne 0 ]; then
		echo "Could not checkout nGraph version $NGRAPH_VERSION"
		exit 1
	fi

	# make ngraph build directory if it doesn't exist
	NGRAPH_BUILD_PATH=$NGRAPH_CLONE_PATH/build
	if [ ! -d $NGRAPH_BUILD_PATH ]; then
		mkdir -p $NGRAPH_BUILD_PATH
		if [ $? -ne 0 ]; then
			echo "Could not create nGraph build directory $NGRAPH_BUILD_PATH"
		       	exit 1
		fi
	fi

	cd $NGRAPH_BUILD_PATH

	cmake ../ -DNGRAPH_INSTALL_PREFIX=$NGRAPH_INSTALL_PATH -DNGRAPH_USE_PREBUILT_LLVM=TRUE
	if [ $? -ne 0 ]; then
		echo "Could not cmake nGraph"
		exit 1
	fi

	make -j$(nproc)
	if [ $? -ne 0 ]; then
		echo "Could not make nGraph"
		exit 1
	fi

	make install
	if [ $? -ne 0 ]; then
		echo "Could not install nGraph"
		exit 1
	fi
fi
