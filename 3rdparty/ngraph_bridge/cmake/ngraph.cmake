#*******************************************************************************
# Copyright 2018 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")


set(NGRAPH_EXTRA_CMAKE_FLAGS "")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_INSTALL_PREFIX=${NGRAPH_INSTALL_PREFIX}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_USE_PREBUILT_LLVM=${NGRAPH_USE_PREBUILT_LLVM}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_UNIT_TEST_ENABLE=0")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_TOOLS_ENABLE=0")

if(USE_NGRAPH_IE)
  add_definitions(-DMXNET_USE_NGRAPH_IE=1)
endif(USE_NGRAPH_IE)

if(USE_NGRAPH_DISTRIBUTED)
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
  link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  add_definitions(-DMXNET_USE_NGRAPH_DISTRIBUTED=1)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_DISTRIBUTED_ENABLE=1")
endif(USE_NGRAPH_DISTRIBUTED)

ExternalProject_Add(
	ext_ngraph
	GIT_REPOSITORY https://github.com/NervanaSystems/ngraph.git
	GIT_TAG abff494

	PREFIX ngraph
	UPDATE_COMMAND ""
	CMAKE_ARGS "${NGRAPH_EXTRA_CMAKE_FLAGS}"
)

set(NGRAPH_INCLUDE_DIR ${NGRAPH_INSTALL_PREFIX}/include)
set(NGRAPH_LIB_DIR ${NGRAPH_INSTALL_PREFIX}/lib)

include_directories(${NGRAPH_INCLUDE_DIR})
