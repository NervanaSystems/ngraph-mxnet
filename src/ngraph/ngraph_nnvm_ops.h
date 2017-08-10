#ifndef NGRAPH_NNVM_OP_H_
#define NGRAPH_NNVM_OP_H_

#include "ngraph_graph.h"

namespace ngraph {

    void register_subgraph(std::shared_ptr<Graph> graph);
} // end ngraph namespace
#endif  // NGRAPH_NNVM_OP_H
