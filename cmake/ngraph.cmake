include(ExternalProject)

add_library(ngraph_interface_lib INTERFACE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")

set(NGRAPH_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/ngraph)
set(NGRAPH_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/ngraph)

set(NGRAPH_EXTRA_CMAKE_FLAGS "")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_INSTALL_PREFIX=${NGRAPH_INSTALL_PREFIX}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_USE_PREBUILT_LLVM=1")

#if(USE_MKLDNN)
#  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DMKLDNN_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/3rdparty/mkldnn/include")
#  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DMKLDNN_LIB_DIR=${CMAKE_BINARY_DIR}/3rdparty/mkldnn/src")
#endif(USE_MKLDNN)

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

message("${NGRAPH_EXTRA_CMAKE_FLAGS}")
ExternalProject_Add(
	ext_ngraph
	SOURCE_DIR ${NGRAPH_SOURCE_DIR}

	PREFIX 3rdparty/ngraph/
	CMAKE_ARGS "${NGRAPH_EXTRA_CMAKE_FLAGS}"

	BUILD_BYPRODUCTS ${NGRAPH_CMAKE_PREFIX}
	BUILD_ALWAYS 1
)

add_dependencies(ngraph_interface_lib ext_ngraph)
if(USE_MKLDNN)
    add_dependencies(ext_ngraph mkldnn)
else()
  include_directories(
    ${NGRAPH_INSTALL_PREFIX}/src/ext_ngraph-build/mkldnn/src/external/mkl/include
  )
endif(USE_MKLDNN)

ExternalProject_Get_Property(ext_ngraph SOURCE_DIR INSTALL_DIR)

target_include_directories(ngraph_interface_lib SYSTEM INTERFACE
	${NGRAPH_INSTALL_PREFIX}/include
)

target_link_libraries(ngraph_interface_lib INTERFACE
	${NGRAPH_INSTALL_PREFIX}/lib/libngraph.so
    ${NGRAPH_INSTALL_PREFIX}/lib/libmkldnn.so
    ${NGRAPH_INSTALL_PREFIX}/lib/libiomp5.so
    ${NGRAPH_INSTALL_PREFIX}/lib/libcpu_backend.so
    ${NGRAPH_INSTALL_PREFIX}/lib/libcodegen.so
    ${NGRAPH_INSTALL_PREFIX}/lib/libmklml_intel.so
)

set(NGRAPH_INCLUDE_DIR ${NGRAPH_INSTALL_DIR}/include)
