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

#include "ngraph_sgcompiler.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_sgcompiler_utils.h"

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

void SGCompiler::ClearOpMap() {
  // delete the temporary storage
  op_map_.clear();
  placeholder_order_.clear();
}

// Compile a Subgraph into ngraph forward and backward call frames
void SGCompiler::CompileSubgraph(std::shared_ptr<Graph> sub_graph) {
  // initalize a placeholder order vector for this subgraph
  for (auto i : sub_graph->inputs_) placeholder_order_.push_back(i);

  // compile all the ndoes in the graph
  for (auto node : sub_graph->nodes_) CompileNode(node, sub_graph);

  // map the inputs into a parameter list
  // TODO: std::transform?
  ngraph::op::Parameters parameters;
  for (auto input : placeholder_order_)
    parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map_[input]));

  // calcuate the shape and return type of the subgraph
  auto shape = TShape_to_NShape(sub_graph->nodes_.back()->shape_);
  auto return_type = std::make_shared<ngraph::TensorViewType>(
      getType(sub_graph->nodes_.back()->dtype_), shape);

  // create the Function object representing the graph
  auto Y = op_map_[sub_graph->nodes_.back()];
  auto f = std::make_shared<ngraph::Function>(Y, return_type, parameters);

  ngraph::traverse_nodes(f, [&sub_graph](std::shared_ptr<ngraph::Node> node) {
    auto param = std::make_shared<ngraph::op::Parameter>(
        node->get_element_type(), node->get_shape());
    sub_graph->fprop_cache.nodes_to_params.Add(node, param);
    sub_graph->fprop_cache.input_params.push_back(param);
    sub_graph->fprop_cache.output_nodes.push_back(node);
  });
  sub_graph->fprop_cache.values.resize(
      sub_graph->fprop_cache.output_nodes.size());

  auto outTuple =
      std::make_shared<ngraph::op::Tuple>(sub_graph->fprop_cache.output_nodes);
  auto outTupleType = outTuple->get_value_type();
  auto newf =
      std::make_shared<ngraph::Function>(outTuple, outTupleType, parameters);

  // compile it into a call frame with the backend, and save
  // the compile frame into the subgraph
  auto forward_external = sub_graph->manager_->compile(newf);
  sub_graph->ngraph_forward =
      sub_graph->backend_->make_call_frame(forward_external);

  // Compile the backward Pass
  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_value_type());

  std::vector<NgraphNodePtr> dYdXs(parameters.size());
  transform(parameters.begin(), parameters.end(), dYdXs.begin(),
            [C, Y](const NgraphNodePtr& X) { return Y->backprop_node(X, C); });

  auto result = std::make_shared<ngraph::op::Tuple>(dYdXs);

  sub_graph->fprop_cache.input_params.insert(
      sub_graph->fprop_cache.input_params.begin(), C);
  auto bf = std::make_shared<ngraph::Function>(
      result, result->get_value_type(), sub_graph->fprop_cache.input_params);

  auto cbf = ngraph::clone_function(bf, sub_graph->fprop_cache.nodes_to_params);

  auto backward_external = sub_graph->manager_->compile(cbf);
  sub_graph->ngraph_backward =
      sub_graph->backend_->make_call_frame(backward_external);
}

// compiling a node, recursively checking it's inputs
void SGCompiler::CompileNode(NodePtr node,
                             const std::shared_ptr<Graph> sub_graph) {
  if (!op_map_.count(node)) {
    // Loop over the inputs and ensure they've been compile3d
    for (auto input : node->inputs_) {
      if (!op_map_.count(input)) {
        // if it's not in the graph, it's an input, compile it as an input
        if (std::find(sub_graph->nodes_.begin(), sub_graph->nodes_.end(),
                      input) == sub_graph->nodes_.end()) {
          CompileInput(input);
        } else {
          CompileNode(input, sub_graph);
        }
      }
    }
    // use the emitter to compile this node and store it
    op_map_[node] = ngraph_op_funcs_[node->operation_](node);
  }
}

// Compile the inputs
void SGCompiler::CompileInput(NodePtr input) {
  auto shape = TShape_to_NShape(input->shape_);
  // make a shaped and typed parameter based on the input node
  // store it in the op_map_
  op_map_[input] =
      std::make_shared<ngraph::op::Parameter>(getType(input->dtype_), shape);
}

}  // namespace ngraph_bridge
