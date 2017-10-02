#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_sgcompiler.h"


namespace ngraph_bridge {

// Main compilation function
std::shared_ptr<Graph> SGCompiler::Compile(NodePtr sub_graph) {
  // clear the op_map and placeholder_order
  ClearOpMap();
  // cast the graph
  auto sg = std::dynamic_pointer_cast<Graph>(sub_graph);
  // compile the subgraph into a python computation
  CompileSubgraph(sg);
  ClearOpMap();
  return sg;
}

void SGCompiler::ClearOpMap(){
  op_map.clear();
  placeholder_order.clear();
}

// Compile a Subgraph into ngraph python objects
void SGCompiler::CompileSubgraph(std::shared_ptr<Graph> sub_graph) {
  // initalize a placeholder order vector for this subgraph
  for (auto i : graph->inputs) placeholder_order.push_back(i->name);

  // Not yet Implemented for ngraph++
  
}

// compiling a node
void SGCompiler::CompileNode(NodePtr node) {
  // if the node has been compiled, return
  if (op_map.count(node->name) > 0) {
    return;
  } else if (NgraphLayerOps_.count(node->operation) != 0) {

  } else if (node->inputs.size() == 1) {

  } else if (node->inputs.size() == 2) {

  } else {
    std::cout << ("operation not yet supported") << std::endl;
    throw;
  }
}

// Compile the inputs, need to implement with ngraph++ axes
void SGCompiler::CompileInput(NodePtr input) {
}

void SGCompiler::CompileInputs(NodePtr node) {
}

}  // end namespace ngraph
