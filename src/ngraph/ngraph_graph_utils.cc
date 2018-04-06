/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <fstream>
#include <iostream>

#include "ngraph_graph.h"
#include "ngraph_graph_utils.h"

namespace ngraph_bridge {

void WriteDot(const Graph& graph, const std::string& fname) {
  // open file stream, write graphviz header
  std::ofstream dotfile;
  dotfile.open(fname);
  dotfile << "digraph G { " << std::endl;
  dotfile << "size=\"8,10.5\"" << std::endl;

  GraphVisitor visitor;

  std::unordered_set<NodePtr> visited;
  // save nodes that match some function condition
  visitor.operation = [&dotfile, &visited](NodePtr node) {
    if (visited.count(node)) {
      return;
    } else {
      visited.insert(node);
    }
    for (auto i : node->inputs_) {
      dotfile << i->name_ << i.get() << " -> " << node->name_ << node.get()
              << ";" << std::endl;
    }
    // write label
    dotfile << node->createNodeLabel() << std::endl;
  };

  visitor.stop_condition = [&visited, &graph](NodePtr node, NodePtr input) {
    // continue if...
    // 2) input not visited
    if (!visited.count(input) && in_vec(graph.nodes_, input)) {
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  for (auto node : graph.outputs_) GraphTraverse(node, visitor);
  for (auto node : graph.inputs_)
    dotfile << node->createNodeLabel() << std::endl;

  // Finish file.
  dotfile << "}" << std::endl;
  dotfile.close();
}

void WriteSubgraphDots(const Graph& graph, const std::string& base) {
  WriteDot(graph, base + ".dot");
  for (auto n : graph.nodes_) {
    if (n->type_ == NodeType::kGraph) {
      auto sg = std::dynamic_pointer_cast<Graph>(n);
      std::ostringstream stream;
      stream << base << sg->subgraph_ << ".dot";
      WriteDot(*sg, stream.str());
    }
  }
}
}  // namespace ngraph_bridge
