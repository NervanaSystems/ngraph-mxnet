cmake_minimum_required(VERSION 3.0.2)


SET(CMAKE_CXX_STANDARD 11)

include(cmake/ngraph.cmake)

include_directories(src src/ops)
include_directories(../../include/)
include_directories(../mshadow/)
include_directories(../dmlc-core/include/)
include_directories(../tvm/nnvm/include/)
include_directories(../tvm/dlpack/include/)
include_directories(build/include/)

add_subdirectory(src)
message("${SRC}")

add_dependencies(ngraph_bridge ext_ngraph)
