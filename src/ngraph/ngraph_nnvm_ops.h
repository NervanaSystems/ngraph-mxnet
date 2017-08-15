#ifndef NGRAPH_NNVM_OP_H_
#define NGRAPH_NNVM_OP_H_

#include "ngraph_graph.h"
#include <nnvm/op.h>

namespace ngraph {
// function for returning nnvm::Op corresponding to a subgraph
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph);
// function for registering subgraph operation with nnvm
void register_subgraph(std::shared_ptr<Graph> graph);

// dummy parameter struct to match mxnet API
struct NGraphParam{
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  void Init(const nnvm::NodeAttrs& attrs){};
};

} // end ngraph namespace
#endif  // NGRAPH_NNVM_OP_H
