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

#include "ngraph_graph.h"
#include "ngraph_graph_utils.h"
#include "reverse_iterate.h"
#include <functional>
#include <stack>

namespace ngraph_bridge {
// Type Aliases
using OpNodePtr = std::shared_ptr<OpNode>;

// Utility for writing a graph to a file for graphviz visualization
void Graph::WriteDot(const std::string& fname) {
  // open file stream, write graphviz header
  std::ofstream dotfile;
  dotfile.open(fname);
  dotfile << "digraph G { " << std::endl;
  dotfile << "size=\"8,10.5\"" << std::endl;

  // Loop over inputs, write graph connections
  for (auto n : nodes_) 
    for (auto i : n->inputs_) {
      if (i->name_ == "") i->name_ = randomString(6);
      if (n->name_ == "") n->name_ = randomString(6);
      dotfile << i->name_ << " -> " << n->name_ << ";" << std::endl;
    }
  // Loop over nodes and write labels
  for (auto n : nodes_) 
    if (!n->name_.empty() )
      dotfile << n->createNodeLabel() << std::endl;
  // Finish file.
  dotfile << "}" << std::endl;
  dotfile.close();
}

// Utility to mark a node as visited and recursive search based on the results
// of an input function
void Graph::DFSUtil(NodePtr s, std::unordered_set<NodePtr>& visited,
                    std::vector<NodePtr>& outNodes,
                    std::function<bool(NodePtr)>& func) {
  // Mark the current node as visited
  visited.insert(s);
  // if this node matches func condition
  if (func(s)) {
    // add it to the output
    outNodes.push_back(s);
    // visit it's inputs
    for (auto i : s->inputs_) {
      if (!visited.count(i) && i->subgraph_ == 0) {
        DFSUtil(i, visited, outNodes, func);
      }
    }
  }
}

// Depth first selection of nodes based on function criterion
std::vector<NodePtr> Graph::DFSselect(NodePtr s,
                                      std::function<bool(NodePtr)> func) {
  // init visited vector
  std::unordered_set<NodePtr> visited;
  // init output vector
  std::vector<NodePtr> outNodes;
  // recursiveliy search the graph
  DFSUtil(s, visited, outNodes, func);
  return outNodes;
}

// Utility for removing bad branches in a directed, acylic subraph.
// Will fail for cyclic graphs
void Graph::RemoveUtil(
    NodePtr s, std::vector<NodePtr>& outNodes,
    std::function<bool(NodePtr)> func,
    std::set<edgeRemoveTup>& visited_edges) {
  // if this node matches func condition
  if (!func(s))
    outNodes.erase(std::remove(outNodes.begin(), outNodes.end(), s),
                   outNodes.end());
  // visit it's inputs if they're still in the subgraph
  for (auto i : s->inputs_)
    if (std::find(outNodes.begin(), outNodes.end(), i) != outNodes.end()) {
      auto edge_tup = edgeRemoveTup{s, i, func(s)};
      if (!visited_edges.count(edge_tup)) {
        visited_edges.insert(edge_tup);
        RemoveUtil(i, outNodes, func, visited_edges);
      }
    }
}

std::vector<NodePtr> Graph::RemoveBroken(NodePtr s,
                                         std::vector<NodePtr>& subgraph_nodes,
                                         std::function<bool(NodePtr)> func) {
  // if this node matches func condition

  std::vector<NodePtr> outNodes;
  std::set<edgeRemoveTup> visited_edges;

  std::function<void(NodePtr)> get_inputs;
  get_inputs = [&outNodes, &get_inputs](NodePtr s) {
    if (std::find(outNodes.begin(), outNodes.end(), s) == outNodes.end()) {
      outNodes.emplace_back(s);
    }
    for (auto i : s->inputs_)
      if (std::find(outNodes.begin(), outNodes.end(), i) == outNodes.end()) {
        get_inputs(i);
      }
  };
  get_inputs(s);

  bool found_bad = false;
  auto good_subgraph_node = [subgraph_nodes, func,
                             found_bad](NodePtr s) mutable {
    if (!func(s)) found_bad = true;
    if (found_bad) return false;
    if (std::find(subgraph_nodes.begin(), subgraph_nodes.end(), s) !=
        subgraph_nodes.end()) {
      return true;
    } else {
      return false;
    }
  };

  RemoveUtil(s, outNodes, good_subgraph_node, visited_edges);
  return outNodes;
}

// I'm not totally sure this is the right approach, but it seems to work.
// Possibly too slow for more complex graphs like DS2
std::vector<NodePtr> Graph::PruneSubgraphOutputs(
    NodePtr s, std::vector<NodePtr>& subgraph_nodes,
    std::function<bool(NodePtr)> func) {
  auto in_graphvec = [](std::vector<NodePtr>& subgraph_nodes,
                        NodePtr s) -> bool {
    if (std::find(subgraph_nodes.begin(), subgraph_nodes.end(), s) ==
        subgraph_nodes.end()) {
      return false;
    } else {
      return true;
    }
  };

  auto get_subgraph_outputs = [this, &subgraph_nodes, &in_graphvec]() {
    std::vector<NodePtr> outNodes;
    for (auto n : nodes_)
      if (!in_graphvec(subgraph_nodes, n))
        for (auto i : n->inputs_)
          if (in_graphvec(subgraph_nodes, i) && !in_graphvec(outNodes, i))
            outNodes.emplace_back(i);
    return outNodes;
  };

  auto prune_subgraph = [&subgraph_nodes](std::vector<NodePtr> outNodes) {
    for (auto n : outNodes)
      if (n != subgraph_nodes[0])
        subgraph_nodes.erase(
            std::remove(subgraph_nodes.begin(), subgraph_nodes.end(), n),
            subgraph_nodes.end());
  };

  std::vector<NodePtr> outNodes;
  bool single_output = false;
  int count = 0;
  while (!single_output && count < 10) {
    outNodes = get_subgraph_outputs();
    if (outNodes.size() <= 1) {
      single_output = true;
    } else {
      prune_subgraph(outNodes);
      subgraph_nodes = RemoveBroken(s, subgraph_nodes, func);
    }
    count += 1;
  }

  return subgraph_nodes;
}


// Find a subgraph, check it for bad branches
std::vector<NodePtr> Graph::FindSubgraph(NodePtr s,
                                         std::function<bool(NodePtr)> func) {
  auto subgraph_nodes = DFSselect(s, func);
  std::vector<NodePtr> outNodes;
  outNodes = subgraph_nodes;
  if (subgraph_nodes.size() > 2) {
    // search for broken loops
    // remove nodes on broken loops
    outNodes = RemoveBroken(s, outNodes, func);
    outNodes = PruneSubgraphOutputs(s, outNodes, func);
  } 
  return outNodes;
}

// function to identify and label connected ngraph ops as subgraphs
void Graph::IdentifySubgraphs(std::function<bool(NodePtr)> func) {
  int sg = 1;
  // loop over the nodes from the back
  for (auto i : reverse_iterate(nodes_)) {
    if (i->subgraph_ == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto subgraph_nodes = FindSubgraph(i, func);
      // if we found a significantly large subgraph, label it
      if (subgraph_nodes.size() > 2) {
        for (auto node : subgraph_nodes)
          node->subgraph_ = sg;
        for (auto node : subgraph_nodes)
          for (auto i : node->inputs_) 
            if (i->subgraph_ != sg)
              i->subgraph_ = -1;
        sg += 1;
      }
    }
  }
}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void Graph::CollapseSubgraphs() {
  // loop variable for undefined number of subgraphs
  int i = 1;
  while (true) {
    auto tmpGraph = std::make_shared<Graph>();
    // loop over all nodes and add nodes in the current subgraph to
    for (auto node : nodes_)
      if (node->subgraph_ == i) tmpGraph->AddNode(node);

    if (tmpGraph->nodes_.size() == 0) {
      // if we don't find any nodes, assume we've run out of subgraphs
      break;
    } else {
      // if we found nodes, setup subgraph
      tmpGraph->in_ngraph_ = true;
      tmpGraph->subgraph_ = i;
      // set node name and shape based on last node in the subgraph
      auto name = tmpGraph->nodes_.back()->name_;
      auto shape = tmpGraph->nodes_.back()->shape_;
      tmpGraph->name_ = "subgraph_" + name + "_" + randomString();
      tmpGraph->shape_ = shape;
      tmpGraph->dtype_ = tmpGraph->nodes_.back()->dtype_;
      auto in_tmpGraphInputs = [&tmpGraph](NodePtr n) {
        if (std::find(tmpGraph->inputs_.begin(), tmpGraph->inputs_.end(), n) ==
            tmpGraph->inputs_.end()) return false;
        return true;
      };
      // setup inputs to this subgraph (as a node)
      for (auto node : tmpGraph->nodes_) {
        for (auto input : node->inputs_) {
          if (input->subgraph_ != i && !in_tmpGraphInputs(input)) 
            tmpGraph->inputs_.emplace_back(input);
        }
      }
      // set subgraph as input to all of the nodes downline.
      for (auto n : nodes_)
        for (size_t i = 0; i < n->inputs_.size(); ++i)
          if (n->inputs_[i]->name_ == name) n->inputs_[i] = tmpGraph;
      
      // find the position we're replacing in the graph
      auto it =
          std::find_if(nodes_.begin(), nodes_.end(),
                       [name](NodePtr n) -> bool { return (n->name_ == name); });
      // insert the subgraph as a node
      nodes_.insert(it, tmpGraph);
    }
    i += 1;
  }

  // delete all the nodes we're replacing with the subgraph
  nodes_.erase(
      std::remove_if(nodes_.begin(), nodes_.end(),
                     [](NodePtr n) -> bool {
                       return ((n->subgraph_ > 0) && 
                               (n->type_ == NodeType::kOp));
                     }),
      nodes_.end());

}

}  // namespace ngraph
