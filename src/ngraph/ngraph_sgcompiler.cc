// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_sgcompiler.h"

namespace ngraph_bridge {

// Main compilation function
std::shared_ptr<Graph> SGCompiler::Compile(NodePtr sub_graph) {
  // clear the op_map_ and placeholder_order
  ClearOpMap();
  // cast the graph
  auto sg = std::dynamic_pointer_cast<Graph>(sub_graph);
  // compile the subgraph into a python computation
  CompileSubgraph(sg);

  return sg;
}

void SGCompiler::ClearOpMap(){
  op_map_.clear();
  placeholder_order_.clear();
}

// Compile a Subgraph into ngraph python objects
void SGCompiler::CompileSubgraph(std::shared_ptr<Graph> sub_graph) {
  // initalize a placeholder order vector for this subgraph
  for (auto i : sub_graph->inputs_) placeholder_order_.push_back(i);

  for (auto node : sub_graph->nodes_) CompileNode(node, sub_graph);
  
  auto manager = ngraph::runtime::Manager::get("NGVM");
  auto backend = manager->allocate_backend();
  
  // Compile the forward Pass
  ngraph::op::Parameters forward_parameters;

  for (auto input : placeholder_order_)
    forward_parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map_[input]));

  auto shape = TShape_to_NShape(sub_graph->nodes_.back()->shape_);
  auto return_type = std::make_shared<ngraph::TensorViewType>(
      getType(sub_graph->nodes_.back()->dtype_), shape);

  auto f = std::make_shared<ngraph::Function>(op_map_[sub_graph->nodes_.back()],
                                              return_type, forward_parameters);

  auto forward_external = manager->compile(f);
  sub_graph->ngraph_forward = backend->make_call_frame(forward_external);


  //Compile the backward Pass
  auto Y = f->get_result();
  auto backward_parameters = forward_parameters;

  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_value_type());

  std::vector<NgraphNodePtr> dYdXs(backward_parameters.size());
  transform(backward_parameters.begin(), backward_parameters.end(),
            dYdXs.begin(), [C, Y](const NgraphNodePtr& X) {
              return Y->backprop_node(X, C);
            });

  auto result = std::make_shared<ngraph::op::Tuple>(dYdXs);
  backward_parameters.insert(backward_parameters.begin(), C);
  auto bf = std::make_shared<ngraph::Function>(result, result->get_value_type(),
                                            backward_parameters);

  auto backward_external = manager->compile(bf);
  sub_graph->ngraph_backward = backend->make_call_frame(backward_external);
}

// compiling a node, recursively checking it's inputs
void SGCompiler::CompileNode(NodePtr node,
                             const std::shared_ptr<Graph> sub_graph) {
  if (!op_map_.count(node)){
    for (auto input : node->inputs_) {
      if (!op_map_.count(input)){
        if (std::find(sub_graph->nodes_.begin(), sub_graph->nodes_.end(),
                      input) == sub_graph->nodes_.end()) {
          CompileInput(input);
        } else {
          CompileNode(input, sub_graph);
        }
      }
    }
    op_map_[node] = ngraph_op_funcs_[node->operation_](node);
  }
}

// Compile the inputs
void SGCompiler::CompileInput(NodePtr input) {
  auto shape = TShape_to_NShape(input->shape_);
  op_map_[input] = std::make_shared<ngraph::op::Parameter>(
      getType(input->dtype_), shape);
}

}

