include(ExternalProject)

add_library(ngraph_interface_lib INTERFACE)

set(EXTERNAL_NGRAPH_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ngraph)
set(EXTERNAL_NGRAPH_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/ngraph)


ExternalProject_Add(
	ext_ngraph
	SOURCE_DIR ${EXTERNAL_NGRAPH_SOURCE_DIR}

	PREFIX 3rdparty/ngraph/
	CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
			-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
			-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
			-DCMAKE_INSTALL_PREFIX=${EXTERNAL_NGRAPH_INSTALL_PREFIX}

	
	BUILD_BYPRODUCTS ${NGRAPH_CMAKE_PREFIX}
	BUILD_ALWAYS 1
)

add_dependencies(ngraph_interface_lib ext_ngraph)

ExternalProject_Get_Property(ext_ngraph SOURCE_DIR INSTALL_DIR)

target_include_directories(ngraph_interface_lib SYSTEM INTERFACE
	${EXTERNAL_NGRAPH_INSTALL_PREFIX}/include
)

target_link_libraries(ngraph_interface_lib INTERFACE
	${INSTALL_DIR}/lib/libngraph.so
    ${INSTALL_DIR}/lib/libmkldnn.so
    ${INSTALL_DIR}/lib/libiomp5.so
    ${INSTALL_DIR}/lib/libcpu_backend.so
    ${INSTALL_DIR}/lib/libcodegen.so
    ${INSTALL_DIR}/lib/libmklml.so
)

set(NGRAPH_INCLUDE_DIR ${EXTERNAL_NGRAPH_INSTALL_DIR}/include)

install(DIRECTORY
	${EXTERNAL_NGRAPH_INSTALL_PREFIX}/lib/
	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
