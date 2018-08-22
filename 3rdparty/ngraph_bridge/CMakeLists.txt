cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
set(NGRAPH_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})

SET(CMAKE_CXX_STANDARD 11)

if(UNIX)
	find_program(LSB_RELEASE_EXEC lsb_release)
	execute_process(COMMAND ${LSB_RELEASE_EXEC} -d
	    OUTPUT_VARIABLE LSB_RELEASE_ID
	    OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	if(LSB_RELEASE_ID MATCHES "Ubuntu 16.04")
		set(NGRAPH_USE_PREBUILT_LLVM 1)
	endif()
endif()


include(cmake/ngraph.cmake)

include_directories(src src/ops)
include_directories(../../include/)
include_directories(../mshadow/)
include_directories(../dmlc-core/include/)
include_directories(../tvm/nnvm/include/)
include_directories(../tvm/dlpack/include/)
include_directories(build/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(src)
message("${SRC}")

add_dependencies(ngraph_bridge ext_ngraph)
