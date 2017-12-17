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
#include <stack>

#include "ngraph_graph.h"
#include "reverse_iterate.h"

namespace ngraph_bridge {
// Type Aliases

/**
* Perform a DFS or Brute graph traversal non-recursively but always ensuring
* that the inputs to a node are operated on before the node.
**/
void GraphTraverse(NodePtr node, const GraphVisitor &visitor, bool DFS) {
  std::unordered_set<NodePtr> visited;  // Mark node as visited
  std::unordered_set<NodePtr> queued;   // Mark node as in the queue
  std::deque<NodePtr> stack;            // Queue of ndoes to process

  // start the stack
  stack.push_front(node);
  queued.insert(node);
  // enter a for loop to process the stack
  while (stack.size() > 0) {
    // get the current node
    auto n = stack.front();

    // if we've visited it, ignore this node
    if (DFS && visited.count(n)) {
      stack.pop_front();
      continue;
    }
    // check the inputs, visit them if
    bool pushed = false;
    for (auto i : visitor.get_inputs(n)) {
      if (!visited.count(i) && !visitor.stop_condition(n, i)) {
        if (!queued.count(i)) {
          // if this is an unvisited, non-stop node, add it to the queue and
          // stop
          stack.push_front(i);
          queued.insert(i);
          pushed = true;
          break;
        } else {
          // if we find a loop in the graph, throw an error
          throw "NGRAPH_BRIDGE: GraphTraverse - This Graph has Cylic Loops!";
        }
      }
    }

    // we want to process inputs before processing a node, so if we've added an
    // input, continue the while loop and process that input
    if (pushed) continue;

    // if we've processed all of th inputs, process the node and remove it from
    // the stack.
    if (DFS) visited.insert(n);
    visitor.operation(n);
    queued.erase(n);

    stack.pop_front();
  }
}
/**
* This utility gets a list of simply-connected nodes that all match some
* function criterion starting from a given node.
**/
std::vector<NodePtr> SelectNodes(NodePtr node,
                                 std::function<bool(NodePtr)> func) {
  // init output vector
  std::vector<NodePtr> outNodes;

  GraphVisitor visitor;

  // the operation of this traversal is to save nodes that match some function
  // condition to the outNodes vector
  visitor.operation = [&outNodes, &func](NodePtr node) {
    if (func(node)) outNodes.push_back(node);
  };

  // if the current node does not match the function condition, we stop
  // searching
  visitor.stop_condition = [&func](NodePtr node, NodePtr input) {
    if (func(node)) {
      return false;
    } else {
      return true;
    }
  };

  // perform the traversal
  DFSGraphTraverse(node, visitor);

  return outNodes;
}

/**
 * This function searches for non-local issues that make parts of an
 * ngraph identified subgraph non-computable
 */
std::vector<NodePtr> RemoveBroken(NodePtr node,
                                  const std::vector<NodePtr> &subgraph_nodes) {
  // create storage for the ouputs and the visited nodes
  std::vector<NodePtr> outNodes;

  /****************************************************************************/
  // First Graph pass - get the intersection of all inputs and outputs of the
  // Current subgraph
  /****************************************************************************/
  GraphVisitor visitor;
  std::unordered_map<NodePtr, bool> is_output;

  visitor.operation = [&outNodes, &is_output, &subgraph_nodes](NodePtr node) {
    // assume this node isn't in the set
    is_output[node] = false;
    // if it's in the subgraph or it's inputs are, mark it as in the set
    if (in_vec(subgraph_nodes, node)) {
      is_output[node] = true;
    } else {
      for (auto input : node->inputs_)
        if (is_output[input]) is_output[node] = true;
    }
    // if this node is in in the set, store it in the output
    if (is_output[node]) outNodes.push_back(node);
  };

  DFSGraphTraverse(node, visitor);

  /****************************************************************************/
  // Second Graph pass - Removing Broken Branches with a Brute search
  /****************************************************************************/

  // First set up a map to check if a node is good or not, mark all subgraph
  // nodes as good.
  std::unordered_map<NodePtr, bool> is_good;
  for (auto n : outNodes) is_good[n] = false;
  for (auto n : subgraph_nodes) is_good[n] = true;

  // The operation of this graph pass is to erase nodes in bad branches
  // from the output
  visitor.operation = [&subgraph_nodes, &is_good, &outNodes](NodePtr node) {
    if (!is_good[node]) {
      outNodes.erase(std::remove(outNodes.begin(), outNodes.end(), node),
                     outNodes.end());
    }
  };

  using EdgeRemoveTuple = std::tuple<NodePtr, bool>;
  std::set<EdgeRemoveTuple> visited_edges;
  // The stop condition of this pass is to check weather
  // 1) if the input to visit is still in the outputs
  // 2) if we've already visiting this input with the current condition
  visitor.stop_condition = [&visited_edges, &outNodes, &is_good](
      NodePtr node, NodePtr input) {
    // if this node is still in the branch
    if (in_vec(outNodes, input)) {
      // if the current node is bad, mark the input as bad so it will be removed
      if (!is_good[node]) is_good[input] = false;
      // check if we've visited it's inputs before with this condition
      auto edge_tup = EdgeRemoveTuple{input, is_good[input]};
      // if we haven't, visit
      if (!visited_edges.count(edge_tup)) {
        visited_edges.insert(edge_tup);
        return false;
      }
    }
    // else, stop traversing the graph
    return true;
  };

  // Remove the bad branches
  BruteGraphTraverse(node, visitor);

  return outNodes;
}

/**
 * Modified subgraph to only return 1 output.
 * If we improve the subgraph compiler/nnvm op construction
 * we might be able to get rid of this pass
 * This removes mutiple outputs from a graph, because the subgraph compiler
 * doesn't currently support multiple outputs
 * TODO: make the subgraph compiler handle multiple outputs and get rid of
 * this
 * graph pass
 */
std::vector<NodePtr> PruneSubgraphOutputs(
    Graph &graph, NodePtr node, std::vector<NodePtr> &subgraph_nodes) {
  // function to get all the outputs of the subgraph
  auto get_subgraph_outputs = [&graph, &subgraph_nodes]() {
    std::vector<NodePtr> outNodes;
    for (auto n : graph.nodes_)
      if (!in_vec(subgraph_nodes, n))
        for (auto i : n->inputs_)
          if (in_vec(subgraph_nodes, i) && !in_vec(outNodes, i))
            outNodes.emplace_back(i);
    return outNodes;
  };

  // function to remove all of the outputs that aren't the last one
  auto prune_subgraph = [&subgraph_nodes](std::vector<NodePtr> outNodes) {
    for (auto n : outNodes)
      if (n != subgraph_nodes.back())
        subgraph_nodes.erase(
            std::remove(subgraph_nodes.begin(), subgraph_nodes.end(), n),
            subgraph_nodes.end());
  };

  // main pass
  // count is for debugging purposes in case the recursive logic is broken
  std::vector<NodePtr> outNodes;
  bool single_output = false;
  int count = 0;
  while (!single_output && count < 100) {
    // get the current outputs
    outNodes = get_subgraph_outputs();
    if (outNodes.size() <= 1) {
      single_output = true;
    } else {
      // we have more than 1 output, remove them and clean any broken loops
      prune_subgraph(outNodes);
      subgraph_nodes = RemoveBroken(node, subgraph_nodes);
    }
    count += 1;
  }

  return subgraph_nodes;
}

// Find a subgraph, check it for bad branches
std::vector<NodePtr> FindSubgraph(Graph &graph, NodePtr node,
                                  std::function<bool(NodePtr)> func) {
  auto subgraph_nodes = SelectNodes(node, func);
  std::vector<NodePtr> outNodes;
  outNodes = subgraph_nodes;
  // search for broken loops
  // remove nodes on broken loops
  outNodes = RemoveBroken(node, outNodes);
  outNodes = PruneSubgraphOutputs(graph, node, outNodes);
  return outNodes;
}

// function to identify and label connected ngraph ops as subgraphs
void IdentifySubgraphs(Graph &graph, std::function<bool(NodePtr)> func) {
  int sg = 1;
  // loop over the nodes from the back
  for (auto i : reverse_iterate(graph.nodes_)) {
    if (i->subgraph_ == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto subgraph_nodes = FindSubgraph(graph, i, func);
      if (subgraph_nodes.size() > 0) {
        // if we found a significantly large subgraph, label it
        for (auto node : subgraph_nodes) node->subgraph_ = sg;
        for (auto node : subgraph_nodes)
          for (auto i : node->inputs_)
            if (i->subgraph_ != sg) i->subgraph_ = -1;
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
      // if we don't find any nodes, assume we've run out of subgraphs
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
      auto in_tmpGraphInputs = [&tmpGraph](NodePtr n) {
        if (!in_vec(tmpGraph->inputs_, n)) return false;
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
      for (auto n : graph.nodes_)
        for (size_t i = 0; i < n->inputs_.size(); ++i)
          if (n->inputs_[i]->name_ == name) n->inputs_[i] = tmpGraph;

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
