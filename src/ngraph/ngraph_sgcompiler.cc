// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/reshape_elimination.hpp>
#include <ngraph/serializer.hpp>

#include "ngraph_sgcompiler_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

static int fcount = 0;

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

  CompileSubgraph(sg);

  return sg;
}

void CompileForwardBackward(std::shared_ptr<Graph> sub_graph,
                            std::shared_ptr<ngraph::Function> f,
                            std::shared_ptr<ngraph::Function> bf,
                            GraphExeMode exe_mode) {
  const int mode = static_cast<int>(exe_mode);

  auto manager = GetManagerFromContext(sub_graph->context_);
  auto backend = GetBackendFromContext(sub_graph->context_);

  sub_graph->ngraph_backward[mode] =
      backend->make_call_frame(manager->compile(bf));
  sub_graph->ngraph_forward[mode] =
      backend->make_call_frame(manager->compile(f));
}

void OptimizeGraph(std::shared_ptr<ngraph::Function> f) {
  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<ngraph::pass::ReshapeElimination>();

  pass_manager.run_passes(f);
}

// Compile a Subgraph into ngraph forward and backward call frames
void SGCompiler::CompileSubgraph(std::shared_ptr<Graph> sub_graph) {
  // initalize a placeholder order vector for this subgraph
  for (auto i : sub_graph->inputs_) placeholder_order_.push_back(i);

  // compile all the ndoes in the graph
  CompileNodes(sub_graph->nodes_.back(), sub_graph);

  ngraph::op::Parameters parameters;
  ngraph::Nodes param_nodes;

  for (const auto input : placeholder_order_) {
    // get the parameters
    parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map_.at(input)));
    param_nodes.push_back(op_map_.at(input));
  }

  // calcuate the shape and return type of the subgraph
  auto Y = op_map_.at(sub_graph->nodes_.back());

  auto backend = GetBackendFromContext(sub_graph->context_);

  const int mode = static_cast<int>(exe_mode_);

  // build ngraph function outputs based on default and aux nodes
  OpNodePtr op_node =
      std::dynamic_pointer_cast<OpNode>(sub_graph->nodes_.back());
  // default output
  ngraph::Nodes outputs{Y};
  // push additional aux outputs
  if (op_node->config_ && !aux_op_map_.empty()) {
    for (auto aux_node : op_node->config_->AuxNodes()) {
      NgraphNodePtr ngraph_node = aux_op_map_.at(aux_node);
      outputs.push_back(ngraph_node);
      if (sub_graph->enable_fprop_cache)
        sub_graph->cached_aux_values[mode].push_back(
            backend->make_primary_tensor_view(ngraph_node->get_element_type(),
                                              ngraph_node->get_shape()));
    }
  }

  // create the Forward Function object representing the graph
  std::shared_ptr<ngraph::Function> f =
      std::make_shared<ngraph::Function>(outputs, parameters);

  OptimizeGraph(f);

  // Create the Adjoint
  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_element_type(),
                                                   Y->get_shape());

  // Perform autodiff
  std::vector<NgraphNodePtr> dYdXs(parameters.size());
  transform(parameters.begin(), parameters.end(), dYdXs.begin(),
            [C, Y](const NgraphNodePtr &X) { return Y->backprop_node(X, C); });

  // create the backward function
  auto back_parameters = parameters;
  back_parameters.insert(back_parameters.begin(), C);

  auto bf = std::make_shared<ngraph::Function>(dYdXs, back_parameters);

  OptimizeGraph(bf);

  if (ngraph_log_graph) {
    dump_graph(f);
    dump_graph(bf);
  }

  if (sub_graph->enable_fprop_cache) {
    auto fprop_cache = ngraph::cache_fprop(f, bf, {C});

    if (ngraph_log_graph) {
      dump_graph(fprop_cache.fprop);
      dump_graph(fprop_cache.bprop);
    }

    CompileForwardBackward(sub_graph, fprop_cache.fprop, fprop_cache.bprop,
                           exe_mode_);

    for (auto node : fprop_cache.fprop_output_nodes) {
      sub_graph->cached_values[mode].push_back(
          backend->make_primary_tensor_view(node->get_element_type(),
                                            node->get_shape()));
    }

  } else {
    CompileForwardBackward(sub_graph, f, bf, exe_mode_);
  }
}

/**
 * Function to perform a graph pass and compile all of the nodes
 * need to make sure we compile the inputs of a node before the node itself
 **/
void SGCompiler::CompileNodes(NodePtr node,
                              const std::shared_ptr<Graph> sub_graph) {
  GraphVisitor visitor;

  // the operation of this graph traverse compiles the node as
  // an input if it's not part of the subgraph or as an ngraph operation
  // if the node is part of the subrraph
  // we capture this so we can save the outputs to the SGCompiler op_map_
  visitor.operation = [this, &sub_graph](NodePtr node) {
    if (!op_map_.count(node)) {
      // if it's not in the graph, it's an input, compile it as an input
      if (!in_vec(sub_graph->nodes_, node)) {
        this->CompileInput(node);
      } else {
        InitOpConfig(std::dynamic_pointer_cast<OpNode>(node));
        assert(ngraph_op_funcs_.find(node->operation_) !=
               ngraph_op_funcs_.end());
        this->op_map_[node] = this->ngraph_op_funcs_[node->operation_](node);
      }
    }
  };

  std::unordered_set<NodePtr> visited;
  visitor.stop_condition = [&sub_graph, &visited](NodePtr node, NodePtr input) {
    // continue if...
    // 1) node is in subgraph or a subgraph input
    // 2) input not visited
    if (in_vec(sub_graph->nodes_, node) && !visited.count(input)) {
      visited.insert(input);
      return false;
    }
    // else, stop traversing the graph
    return true;
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
