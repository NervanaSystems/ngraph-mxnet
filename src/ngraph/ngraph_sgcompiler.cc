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
#include <iterator>
#include <vector>
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

  std::vector<NgraphNodePtr> results(sub_graph->subgraph_outputs_.size());

  transform(sub_graph->subgraph_outputs_.begin(),
            sub_graph->subgraph_outputs_.end(), results.begin(),
            [this](const NodePtr X) { return this->op_map_[X]; });

  auto result = std::make_shared<ngraph::op::Tuple>(results);
  // create the Function object representing the graph
  auto f = std::make_shared<ngraph::Function>(result, result->get_value_type(),
                                              parameters);

  // compile it into a call frame with the backend, and save
  // the compile frame into the subgraph
  auto forward_external = sub_graph->manager_->compile(f);
  sub_graph->ngraph_forward =
      sub_graph->backend_->make_call_frame(forward_external);

  // Compile the backward Pass
  auto Y = f->get_result();

  std::vector<NgraphNodePtr> dYdXs;
  for (auto rarg : Y->get_arguments()) {
    auto C = std::make_shared<ngraph::op::Parameter>(
        rarg->get_value_type());  // delta per each ouput?
    backward_parameters.push_back(C);
    for (auto X : parameters)
      dYdXs.push_back(rarg->backprop_node(X, C));  // constructing cross product
                                                   // even though not every
                                                   // input is used by every
                                                   // output,
  }  // but ngraph is able to figure this out
  result = std::make_shared<ngraph::op::Tuple>(dYdXs);

  std::copy(backward_parameters.rbegin(), backward_parameters.rend(),
            std::inserter(parameters,
                          begin(parameters)));  // this puts deltas
                                                // for each output node
                                                // in front of parameters
                                                // in the same order as
                                                // output_nodes_

  auto bf = std::make_shared<ngraph::Function>(result, result->get_value_type(),
                                               backward_parameters);

  auto backward_external = sub_graph->manager_->compile(bf);
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
