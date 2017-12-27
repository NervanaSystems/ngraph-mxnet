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
  auto f = std::make_shared<ngraph::XLAFunction>(
      op_map_[sub_graph->nodes_.back()], return_type, parameters);

  if (dump) dump_graph(f);

  // compile it into a call frame with the backend, and save
  // the compile frame into the subgraph
  auto forward_external =
      GetManagerFromContext(sub_graph->context_)->compile(f);
  sub_graph->ngraph_forward = GetBackendFromContext(sub_graph->context_)
                                  ->make_call_frame(forward_external);

  // Compile the backward Pass
  auto Y = f->get_result();

  auto C = std::make_shared<ngraph::op::Parameter>(Y->get_value_type());

  std::vector<NgraphNodePtr> dYdXs(parameters.size());
  transform(parameters.begin(), parameters.end(), dYdXs.begin(),
            [C, Y](const NgraphNodePtr &X) { return Y->backprop_node(X, C); });

  auto result = std::make_shared<ngraph::op::XLATuple>(dYdXs);
  parameters.insert(parameters.begin(), C);
  auto bf = std::make_shared<ngraph::XLAFunction>(
      result, result->get_value_type(), parameters);

  if (dump) dump_graph(bf);

  auto backward_external =
      GetManagerFromContext(sub_graph->context_)->compile(bf);
  sub_graph->ngraph_backward = GetBackendFromContext(sub_graph->context_)
                                   ->make_call_frame(backward_external);
}

// compiling a node, recursively checking it's inputs
void SGCompiler::CompileNodes(NodePtr node,
                              const std::shared_ptr<Graph> sub_graph) {
  CompileNodesGraphVisitor visitor(this, sub_graph);
  DFSGraphTraverse(node, visitor);
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
