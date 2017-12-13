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

#include <functional>
#include <iterator>
#include <stack>
#include <vector>

#include "ngraph_graph.h"
#include "reverse_iterate.h"

namespace ngraph_bridge {
// Type Aliases
using EdgeRemoveTuple = std::tuple<NodePtr, NodePtr, bool>;

/**
 * Utility to mark a node as visited and recursive search based on the results
 * of an input function
 */
void DFSUtil(NodePtr node, std::unordered_set<NodePtr> &visited,
             std::vector<NodePtr> &outNodes,
             std::function<bool(NodePtr)> &func) {
  // Mark the current node as visited
  visited.insert(node);
  // if this node matches func condition
  if (func(node)) {
    // add it to the output
    outNodes.push_back(node);
    // visit it's inputs
    for (auto i : node->inputs_) {
      if (!visited.count(i) && i->subgraph_ == 0) {
        DFSUtil(i, visited, outNodes, func);
      }
    }
  }
}

// Depth first selection of nodes based on function criterion
std::vector<NodePtr> SelectNodes(NodePtr node,
                                 std::function<bool(NodePtr)> func) {
  // init visited vector
  std::unordered_set<NodePtr> visited;
  // init output vector
  std::vector<NodePtr> outNodes;
  // recursiveliy search the graph
  DFSUtil(node, visited, outNodes, func);
  return outNodes;
}

/**
 * Utility for removing bad branches in a directed, acylic subraph.
 * Will fail for cyclic graphs
 */
void RemoveUtil(NodePtr node, std::vector<NodePtr> &outNodes,
                std::function<bool(NodePtr)> func,
                std::set<EdgeRemoveTuple> &visited_edges) {
  // if this node doesn't match the function condition, delete it
  if (!func(node))
    outNodes.erase(std::remove(outNodes.begin(), outNodes.end(), node),
                   outNodes.end());

  // visit it's inputs if they're still in the subgraph
  for (auto i : node->inputs_)
    if (in_vec(outNodes, i)) {
      // ask if we've already gone up this branch in this closure state.
      // if so, don't revisit, if not, try it both good and bad.
      auto edge_tup = EdgeRemoveTuple{node, i, func(node)};
      if (!visited_edges.count(edge_tup)) {
        visited_edges.insert(edge_tup);
        RemoveUtil(i, outNodes, func, visited_edges);
      }
    }
}

/**
 * This function searches for non-local issues that make parts of an
 * ngraph identified subgraph non-computable
 */
std::vector<NodePtr> RemoveBroken(NodePtr node,
                                  std::vector<NodePtr> &subgraph_nodes,
                                  std::function<bool(NodePtr)> func) {
  // create storage for the ouputs and the visited nodes
  std::vector<NodePtr> outNodes;
  std::unordered_set<NodePtr> visited;

  // This function searches the nodes that are inputs to the final
  // subgraph output AND outputs of other subgraph nodes
  // to minimize what needs to be searched for broken loops
  std::function<bool(NodePtr)> get_nodes;
  get_nodes = [&outNodes, &visited, &get_nodes, &func](NodePtr n) {
    visited.insert(n);
    bool im_an_output = false;
    if (func(n)) im_an_output = true;

    for (auto i : n->inputs_) {
      if (!in_vec(outNodes, i)) {
        if (!visited.count(i))
          if (get_nodes(i)) im_an_output = true;
      } else {
        im_an_output = true;
      }
    }

    if (im_an_output) outNodes.push_back(n);
    return im_an_output;
  };

  get_nodes(node);

  // This is a mutable closure, copied on each step up the graph,
  // that tells us weather or not this branch of the graph is good or bad
  bool found_bad = false;
  auto good_subgraph_node = [subgraph_nodes, func,
                             found_bad](NodePtr n) mutable {
    if (!func(n)) found_bad = true;
    if (found_bad) return false;
    if (in_vec(subgraph_nodes, n)) {
      return true;
    } else {
      return false;
    }
  };

  std::set<EdgeRemoveTuple> visited_edges;
  // recursive search for bad branches
  RemoveUtil(node, outNodes, good_subgraph_node, visited_edges);
  return outNodes;
}

// This removes mutiple outputs from a graph, because the subgraph compiler
// doesn't currently support multiple outputs
// TODO: make the subgraph compiler handle multiple outputs and get rid of this
// graph pass
std::vector<NodePtr> PruneSubgraphOutputs(Graph &graph, NodePtr node,
                                          std::vector<NodePtr> &subgraph_nodes,
                                          std::function<bool(NodePtr)> func) {
  std::vector<NodePtr> outNodes;
  for (auto n : graph.nodes_)
    if (!in_vec(subgraph_nodes, n))
      for (auto i : n->inputs_) {
        if (in_vec(subgraph_nodes, i)) {
          if (!in_vec(outNodes, i)) {  // TODO: use set
            //@#$ FIX
            auto index = std::distance(
                begin(outNodes),
                std::find(begin(outNodes), end(outNodes),
                          i));  // if not found index is the no of elements
            if (index == outNodes.size()) {
              outNodes.emplace_back(i);
            }
            n->input_index_ = index;
          }
        }
      }
  return outNodes;
}

// Find a subgraph, check it for bad branches
std::pair<std::vector<NodePtr>, std::vector<NodePtr>> FindSubgraph(
    Graph &graph, NodePtr s, std::function<bool(NodePtr)> func) {
  auto subgraph_nodes = SelectNodes(s, func);
  std::vector<NodePtr> outNodes;
  outNodes = subgraph_nodes;
  // search for broken loops
  // remove nodes on broken loops
  subgraph_nodes = RemoveBroken(s, outNodes, func);
  outNodes = PruneSubgraphOutputs(graph, s, subgraph_nodes, func);
  return std::make_pair(subgraph_nodes, outNodes);
}

// function to identify and label connected ngraph ops as subgraphs
void IdentifySubgraphs(Graph &graph, std::function<bool(NodePtr)> func) {
  int sg = 1;
  graph.subgraphs_outputs_.push_back(
      std::vector<NodePtr>{nullptr});  // since sg counts from 1 and up
  // loop over the nodes from the back
  for (auto i : reverse_iterate(graph.nodes_)) {
    if (i->subgraph_ == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto pair = FindSubgraph(graph, i, func);
      auto subgraph_nodes = pair.first;
      // if we found a significantly large subgraph, label it
      if (subgraph_nodes.size() > 0) {
        auto &outNodes = pair.second;
        for (auto node : subgraph_nodes) node->subgraph_ = sg;
        for (auto node : subgraph_nodes)
          for (auto i : node->inputs_)
            if (i->subgraph_ != sg) {
              auto pos = std::find(begin(outNodes), end(outNodes), i);
              if (pos != end(outNodes)) {
                outNodes.erase(pos);
              }
              i->subgraph_ = -1;  // are any of these in outNodes should we
                                  // replace those w/ nullptr
            }

        graph.subgraphs_outputs_.emplace_back(outNodes);
        sg += 1;
      }
    }
  }
}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void CollapseSubgraphs(Graph &graph) {
  // loop variable for undefined number of subgraphs
  int i = 1;
  while (true) {
    auto tmpGraph = std::make_shared<Graph>(
        "subgraph_" + randomString(12) + std::to_string(i), graph.context_);
    // loop over all nodes and add nodes in the current subgraph to
    for (auto node : graph.nodes_)
      if (node->subgraph_ == i) tmpGraph->AddNode(node);

    if (tmpGraph->nodes_.size() == 0) {
      // if we don't find any nodes, assume we've run out of subgraphs TODO:
      // [nikolayk] how did we get there?
      break;
    } else {
      // if we found nodes, setup subgraph
      tmpGraph->in_ngraph_ = true;
      tmpGraph->subgraph_ = i;
      // set node name and shape based on last node in the subgraph
      auto name = tmpGraph->nodes_.back()->name_;
      auto shape = tmpGraph->nodes_.back()->shape_;
      tmpGraph->shape_ = shape;
      tmpGraph->dtype_ = tmpGraph->nodes_.back()->dtype_;
      auto in_tmpGraphInputs = [&tmpGraph](
          NodePtr
              n) {  //[nikolayk] 2nd time seeing this pattern why not use set?
        if (!in_vec(tmpGraph->inputs_, n)) return false;
        return true;
      };
      // setup inputs to this subgraph (as a node)
      for (auto node : tmpGraph->nodes_) {
        for (auto input : node->inputs_) {
          if (input->subgraph_ != i && !in_tmpGraphInputs(input)) {
            tmpGraph->inputs_.emplace_back(input);
          }
        }
      }

      auto &sgOuts = graph.subgraphs_outputs_[tmpGraph->subgraph_];
      tmpGraph->subgraph_outputs_.insert(end(tmpGraph->subgraph_outputs_),
                                         begin(sgOuts), end(sgOuts));

      // set subgraph as input to all of the nodes downline.
      for (auto n : graph.nodes_)
        for (size_t i = 0; i < n->inputs_.size(); ++i) {
          if (n->inputs_[i]->name_ == name) {
            auto index = std::distance(
                begin(tmpGraph->subgraph_outputs_),
                std::find(begin(tmpGraph->subgraph_outputs_),
                          end(tmpGraph->subgraph_outputs_), n->inputs_[i]));
            if (index == tmpGraph->subgraph_outputs_.size()) {
              throw "NGRAPH_BRIDGE: input node must be in the outputs of the graph"; //assert
            }
            n->inputs_[i] = tmpGraph;
            n->inputs_[i]->input_index_ = index;
          }
        }

      // find the position we're replacing in the graph
      auto it = std::find_if(
          graph.nodes_.begin(), graph.nodes_.end(),
          [name](NodePtr n) -> bool { return (n->name_ == name); });
      // insert the subgraph as a node
      graph.nodes_.insert(it, tmpGraph);
    }
    i += 1;
  }

  // delete all the nodes we're replacing with the subgraph
  graph.nodes_.erase(std::remove_if(graph.nodes_.begin(), graph.nodes_.end(),
                                    [](NodePtr n) -> bool {
                                      return ((n->subgraph_ > 0) &&
                                              (n->type_ == NodeType::kOp));
                                    }),
                     graph.nodes_.end());
}

}  // namespace ngraph_bridge
