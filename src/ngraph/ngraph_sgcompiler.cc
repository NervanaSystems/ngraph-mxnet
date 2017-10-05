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
  for (auto i : sub_graph->inputs) placeholder_order.push_back(i);

  for (auto node : sub_graph->nodes_){
    for (auto input : node->inputs) {
      if (!op_map.count(input)){
        if (std::find(sub_graph->nodes_.begin(), sub_graph->nodes_.end(),
                      input) == sub_graph->nodes_.end()) {
          CompileInput(input);
        } else {
          CompileNode(input);
        }
      }
    }
    CompileNode(node);
  }

  auto manager = ngraph::runtime::Manager::get("NGVM");
  auto backend = manager->allocate_backend();
  
  ngraph::op::Parameters forward_parameters;

  for (auto input : placeholder_order)
    forward_parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map[input]));

  auto shape = TShape_to_NShape(sub_graph->nodes_.back()->shape);
  auto return_type = std::make_shared<ngraph::TensorViewType>(
      getType(sub_graph->nodes_.back()->dtype), shape);

  auto f = std::make_shared<ngraph::Function>(op_map[sub_graph->nodes_.back()],
                                              return_type, forward_parameters);
  
  auto forward_external = manager->compile(f);
  sub_graph->ngraph_forward = backend->make_call_frame(forward_external);
      
}

// compiling a node
void SGCompiler::CompileNode(NodePtr node) {
  // if the node has been compiled, return
  if (op_map.count(node) > 0) {
    return;
  } else if (NgraphLayerOps_.count(node->operation) != 0) {

  } else if (node->inputs.size() == 1) {
    op_map[node] = NgraphUnaryOps_[node->operation](op_map[node->inputs[0]]);
  } else if (node->inputs.size() == 2) {
    op_map[node] = NgraphBinaryOps_[node->operation](op_map[node->inputs[0]],
                                                     op_map[node->inputs[1]]);
  } else {
    std::cout << ("operation not yet supported") << std::endl;
    throw;
  }
}

// Compile the inputs
void SGCompiler::CompileInput(NodePtr input) {
  auto shape = TShape_to_NShape(input->shape);
  op_map[input] = std::make_shared<ngraph::op::Parameter>(
      getType(input->dtype), shape);
}

}

