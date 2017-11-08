//
// Created by Louis Feng on 11/6/17.
//

#include <iostream>
#include <fstream>

#include "ngraph_graph.h"
#include "ngraph_graph_utils.h"

namespace ngraph_bridge {

void WriteDot(GraphPtr& graph, const std::string& fname) {
  // open file stream, write graphviz header
  std::ofstream dotfile;
  dotfile.open(fname);
  dotfile << "digraph G { " << std::endl;
  dotfile << "size=\"8,10.5\"" << std::endl;

  // Loop over inputs, write graph connections
  for (auto n : graph->GetNodes())
    for (auto i : n->inputs_) {
      dotfile << i->name_ << " -> " << n->name_ << ";" << std::endl;
    }
  // Loop over nodes and write labels
  for (auto n : graph->GetNodes())
    if (!n->name_.empty()) dotfile << n->createNodeLabel() << std::endl;
  // Finish file.
  dotfile << "}" << std::endl;
  dotfile.close();
}

void WriteSubgraphDots(const GraphPtr& graph, const std::string &base) {
  WriteDot(graph, base + ".dot");
  for (auto n : graph->GetNodes()) {
    if (n->type_ == NodeType::kGraph) {
      auto sg = std::dynamic_pointer_cast<Graph>(n);
      std::ostringstream stream;
      stream << base << sg->subgraph_ << ".dot";
      WriteDot(sg, stream.str());
    }
  }
}
} // namespace