set(SRC 
    ngraph_compiler.cc
    ngraph_emitter.cc
    ngraph_emitter_utils.cc
    ngraph_graph.cc
    ngraph_graph_utils.cc
    ngraph_imperative.cc
    ngraph_nnvm_ops.cc
    ngraph_sgcompiler.cc
    ngraph_stats.cc
    ngraph_subgraph.cc
    ngraph_utils.cc
    ops/batchnorm.cc
    ops/deconvolution.cc
    ops/slice.cc
    )

add_library(ngraph_bridge OBJECT ${SRC})

if(USE_CUDA)
    add_definitions(-DMSHADOW_USE_CUDA=1)
else()
    add_definitions(-DMSHADOW_USE_CUDA=0)
endif()
if (BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    add_definitions(-DMSHADOW_USE_MKL=1)
    add_definitions(-DMSHADOW_USE_CBLAS=0)
else()
    add_definitions(-DMSHADOW_USE_MKL=0)
    add_definitions(-DMSHADOW_USE_CBLAS=1)
endif()


add_definitions(-DMXNET_USE_NGRAPH=1)
