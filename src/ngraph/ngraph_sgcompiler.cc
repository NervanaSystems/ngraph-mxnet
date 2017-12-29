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
#include <ngraph/serializer.hpp>
#include "ngraph_sgcompiler_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace ngraph_bridge {

static int fcount = 0;

static bool dump = false;

void dump_graph(std::shared_ptr<ngraph::Function> f) {
  std::stringstream fname;
  fname << "Graph_" << fcount << ".json";
  fcount += 1;
  std::ofstream file;
  file.open(fname.str());
  file << ngraph::serialize(f) << std::endl;
  file.close();
}

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
  CompileNodes(sub_graph->nodes_.back(), sub_graph);

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
  auto f = std::make_shared<ngraph::XLAFunction>(Y, return_type, parameters);

  // Create the Backward Pass
  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_value_type());

  // Perform autodiff
  std::vector<NgraphNodePtr> dYdXs(parameters.size());
  transform(parameters.begin(), parameters.end(), dYdXs.begin(),
            [C, Y](const NgraphNodePtr &X) { return Y->backprop_node(X, C); });

  auto result = std::make_shared<ngraph::op::XLATuple>(dYdXs);

  // create the backward function
  auto back_parameters = parameters;
  back_parameters.insert(back_parameters.begin(), C);

  auto bf = std::make_shared<ngraph::XLAFunction>(
      result, result->get_value_type(), back_parameters);

  auto fprop_cache = ngraph::cache_fprop(f, bf, {C});

  if (dump) {
    dump_graph(fprop_cache.fprop);
    dump_graph(fprop_cache.bprop);
  }

  auto manager = GetManagerFromContext(sub_graph->context_);
  auto backend = GetBackendFromContext(sub_graph->context_);

  auto forward_external = manager->compile(fprop_cache.fprop);
  sub_graph->ngraph_forward = backend->make_call_frame(forward_external);

  auto backward_external = manager->compile(fprop_cache.bprop);
  sub_graph->ngraph_backward = backend->make_call_frame(backward_external);

  for (auto node : fprop_cache.fprop_output_nodes) {
    sub_graph->cached_values.push_back(backend->make_primary_tensor_view(
        node->get_element_type(), node->get_shape()));
  }
}

/**
 * Function to perform a graph pass and compile all of the nodes
 * need to make sure we compile the inputs of a node before the node itself
 **/
void SGCompiler::CompileNodes(NodePtr node,
                              const std::shared_ptr<Graph> sub_graph) {
  GraphVisitor visitor;
  std::unordered_set<NodePtr> visited;

  // the operation of this graph traverse compiles the node as
  // an input if it's not part of the subgraph or as an ngraph operation
  // if the node is part of the subrraph
  // we capture this so we can save the outputs to the SGCompiler op_map_
  visitor.operation = [this, &sub_graph, &visited](NodePtr node) {
    visited.insert(node);
    if (!op_map_.count(node)) {
      // if it's not in the graph, it's an input, compile it as an input
      if (!in_vec(sub_graph->nodes_, node)) {
        this->CompileInput(node);
      } else {
        this->op_map_[node] = this->ngraph_op_funcs_[node->operation_](node);
      }
    }
  };

  // Don't compile any nodes that aren't inputs to the subgraph or in the
  // subgraph
  visitor.stop_condition = [&sub_graph, &visited](NodePtr node, NodePtr input) {
    if (in_vec(sub_graph->nodes_, node) && !visited.count(input)) {
      return false;
    } else {
      return true;
    }
  };

  GraphTraverse(node, visitor);
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
