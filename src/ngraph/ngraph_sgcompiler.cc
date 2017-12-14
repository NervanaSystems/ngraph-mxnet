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

  FpropCache fprop_cache;

  ngraph::op::Parameters parameters;
  ngraph::Nodes param_nodes;

  for (auto input : placeholder_order_) {
    // get the parameters
    parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map_[input]));
    param_nodes.push_back(op_map_[input]);
  }
  // calcuate the shape and return type of the subgraph
  auto Y = op_map_[sub_graph->nodes_.back()];
  auto return_type = std::make_shared<ngraph::TensorViewType>(
      Y->get_element_type(), Y->get_shape());

  // create the Forward Function object representing the graph
  auto f = std::make_shared<ngraph::Function>(Y, return_type, parameters);

  // Create the Backward Pass
  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_value_type());
  
  // Perform autodiff
  std::vector<NgraphNodePtr> dYdXs(parameters.size());
  transform(parameters.begin(), parameters.end(), dYdXs.begin(),
            [C, Y](const NgraphNodePtr& X) { return Y->backprop_node(X, C); });

  auto result = std::make_shared<ngraph::op::Tuple>(dYdXs);

  // create the backward function
  auto back_parameters = parameters;
  back_parameters.insert(back_parameters.begin(), C);

  auto bf = std::make_shared<ngraph::Function>(result, result->get_value_type(),
                                               back_parameters);

  // create a map from f nodes to parameters
  std::unordered_map<NgraphNodePtr, NgraphNodePtr> node_param_map;
  ngraph::traverse_nodes(f, [&node_param_map](NgraphNodePtr node) {
    auto param = std::make_shared<ngraph::op::Parameter>(
        node->get_element_type(), node->get_shape());
    node_param_map[node] = param;
  });

  std::unordered_set<NgraphNodePtr> in_bprop;
  ngraph::traverse_nodes(bf, [&in_bprop](NgraphNodePtr node) {
    if (!in_bprop.count(node)) in_bprop.insert(node);
  });

  std::vector<NgraphNodePtr> unused_nodes;
  for (auto kv : node_param_map) {
    if (!in_bprop.count(kv.first) || in_vec(param_nodes, kv.first)) {
      unused_nodes.push_back(kv.first);
    } else {
      fprop_cache.fprop_output_nodes.push_back(kv.first);
    }
  }
  for (auto node : unused_nodes) node_param_map.erase(node);

  for (auto node : fprop_cache.fprop_output_nodes) {
    fprop_cache.values.push_back(sub_graph->backend_->make_primary_tensor_view(
        node->get_element_type(), node->get_shape()));
  }

  std::vector<NgraphNodePtr> fprop_outputs{Y};
  fprop_outputs.insert(fprop_outputs.end(),
                       fprop_cache.fprop_output_nodes.begin(),
                       fprop_cache.fprop_output_nodes.end());
  auto outTuple =
      std::make_shared<ngraph::op::Tuple>(fprop_outputs);
  auto outTupleType = outTuple->get_value_type();
  auto newf =
       std::make_shared<ngraph::Function>(outTuple, outTupleType, parameters);

  
  ngraph::NodeMap node_map;
  for (auto kv : node_param_map) {
    node_map.Add(kv.first, kv.second);
  }

  ngraph::clone_nodes(bf->get_ops(), node_map);

  // get cloned function result and parameters
  auto cloned_result = node_map[bf->get_result()];

  ngraph::op::Parameters cloned_params;
  for (auto param : bf->get_parameters())
  {
      cloned_params.push_back(std::dynamic_pointer_cast<ngraph::op::Parameter>(node_map[param]));
  }

  fprop_cache.bprop_input_params = cloned_params;

  for (auto x : fprop_cache.fprop_output_nodes) {
    fprop_cache.bprop_input_params.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(node_map[x]));
  }

  // create and return cloned function
  auto newbf = std::make_shared<ngraph::Function>(
      cloned_result, cloned_result->get_value_type(), fprop_cache.bprop_input_params);

  sub_graph->fprop_cache = fprop_cache;

  auto forward_external = sub_graph->manager_->compile(newf);
  sub_graph->ngraph_forward =
      sub_graph->backend_->make_call_frame(forward_external);

  auto backward_external = sub_graph->manager_->compile(newbf);
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
