#ifndef NGRAPH_NNVM_OP_H_
#define NGRAPH_NNVM_OP_H_

#include "ngraph_graph.h"
#include <nnvm/op.h>

namespace ngraph {

    nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph);
    void register_subgraph(std::shared_ptr<Graph> graph);
} // end ngraph namespace
#endif  // NGRAPH_NNVM_OP_H
