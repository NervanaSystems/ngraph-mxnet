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

  // Loop over inputs, write graph connections
  for (auto n : graph.nodes_)
    for (auto i : n->inputs_) {
      dotfile << i->name_ << i.get() << " -> " << n->name_ << n.get() << ";"
              << std::endl;
    }
  // Loop over nodes and write labels
  for (auto n : graph.nodes_) dotfile << n->createNodeLabel() << std::endl;
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
