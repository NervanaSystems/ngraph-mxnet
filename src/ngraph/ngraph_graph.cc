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

#include <deque>
#include <functional>
#include <stdexcept>
#include <unordered_set>

#include "ngraph_graph.h"

namespace ngraph_bridge {
// Type Aliases

/**
 * Perform a DFS graph traversal non-recursively but always ensuring
 * that the inputs to a node are operated on before the node.
 **/
void GraphTraverse(NodePtr node, const GraphVisitor& visitor) {
  // faster queue lookup datastructure, slighlty more memory
  // than just using std::find on the stack, but much faster
  // cyclic checking when graphs get large
  std::unordered_set<NodePtr> queued;

  // start the stack
  std::deque<NodePtr> stack;
  stack.push_front(node);
  queued.insert(node);

  // enter loop to process the stack
  while (stack.size() > 0) {
    // get the current node
    auto n = stack.front();
    bool input_pushed = false;
    for (auto i : visitor.get_inputs(n)) {
      // check for cyclic graph
      if (queued.count(i) != 0) {
        throw std::runtime_error(
            "NGRAPH_BRIDGE: GraphTraverse - This Graph has Cylic Loops!");
      }

      // call visitor to determine whether to push input
      if (!visitor.stop_condition(n, i)) {
        stack.push_front(i);
        queued.insert(i);
        input_pushed = true;
        break;  // depth first search
      }
    }

    // we want to process inputs before processing a node, so if we've added an
    // input, continue the while loop and process that input
    if (input_pushed) continue;

    // if we've processed all of the inputs, process the node and remove it from
    // the stack.
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
                                 const std::function<bool(NodePtr)>& func) {
  std::vector<NodePtr> outNodes;

  GraphVisitor visitor;

  // save nodes that match some function condition
  visitor.operation = [&outNodes, &func](NodePtr node) {
    if (node->subgraph_ > 0) {
      return;
    }
    if (func(node)) outNodes.push_back(node);
  };

  std::unordered_set<NodePtr> visited;
  visitor.stop_condition = [&visited, &func](NodePtr node, NodePtr input) {
    // continue if...
    // 1) current node matches function condition
    // 2) input not visited
    if (func(node) && !visited.count(input) && input->subgraph_ < 1) {
      visited.insert(input);
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  GraphTraverse(node, visitor);

  return outNodes;
}

/**
 * This function searches for non-local issues that make parts of an
 * ngraph identified subgraph non-computable
 */
std::vector<NodePtr> RemoveBroken(NodePtr node,
                                  const std::vector<NodePtr>& subgraph_nodes) {
  std::vector<NodePtr> outNodes;

  /****************************************************************************/
  // First Graph pass - get the intersection of all inputs to a subgraph rooted
  // at node and outputs of the nodes in subgraph_nodes
  /****************************************************************************/
  GraphVisitor visitor1;
  std::unordered_map<NodePtr, bool> is_output;

  visitor1.operation = [&outNodes, &is_output, &subgraph_nodes](NodePtr node) {
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

  std::unordered_set<NodePtr> visited1;
  visitor1.stop_condition = [&visited1](NodePtr node, NodePtr input) {
    // continue if input node not visited
    if (!visited1.count(input)) {
      visited1.insert(input);
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  GraphTraverse(node, visitor1);

  /****************************************************************************/
  // Second Graph pass - Removing Broken Branches
  /****************************************************************************/
  GraphVisitor visitor2;

  // track 'good' status of all nodes
  // subgraph nodes (identified by SelectNodes) are 'good' (true)
  // output nodes NOT in subgraph nodes are 'not good' (false)
  std::unordered_map<NodePtr, bool> is_good;
  for (auto n : outNodes) is_good[n] = false;
  for (auto n : subgraph_nodes) is_good[n] = true;

  // erase nodes in bad branches from the output
  visitor2.operation = [&is_good, &outNodes](NodePtr node) {
    if (!is_good[node]) {
      outNodes.erase(std::remove(outNodes.begin(), outNodes.end(), node),
                     outNodes.end());
    }
  };

  // represents an input and the 'good' status of the node it was called from
  using NodeGood = std::tuple<NodePtr, bool>;
  std::set<NodeGood> visited2;

  visitor2.stop_condition = [&visited2, &outNodes, &is_good](NodePtr node,
                                                             NodePtr input) {
    // propagate 'good' status from node to input
    if (!is_good[node]) is_good[input] = false;
    auto edge_tup = NodeGood{input, is_good[input]};

    // continue if...
    // 1) the input is still in output nodes
    // 2) the input has not already been visited with its 'good' status
    if (in_vec(outNodes, input) && !visited2.count(edge_tup)) {
      visited2.insert(edge_tup);
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  GraphTraverse(node, visitor2);

  /****************************************************************************/
  // Third Graph pass - Removing nodes that are no longer connected to the main
  // output
  /****************************************************************************/
  GraphVisitor visitor3;
  std::unordered_map<NodePtr, bool> is_connected;
  for (auto n : outNodes) is_connected[n] = false;
  // erase nodes in bad branches from the output
  visitor3.operation = [&is_connected](NodePtr node) {
    is_connected[node] = true;
  };

  // represents an input and the 'good' status of the node it was called from
  using NodeGood = std::tuple<NodePtr, bool>;
  std::set<NodePtr> visited3;

  visitor3.stop_condition = [&visited3, &outNodes](NodePtr node,
                                                   NodePtr input) {
    // continue if...
    // 1) the input is in output nodes
    // 2) the input has not already been visited
    if (in_vec(outNodes, input) && !visited3.count(input)) {
      visited3.insert(input);
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  GraphTraverse(node, visitor3);

  std::vector<NodePtr> out;
  for (auto node : outNodes) {
    if (is_connected[node]) {
      out.push_back(node);
    }
  }
  return out;
}

std::vector<NodePtr> GetSubgraphOutputs(
    const Graph& graph, const std::vector<NodePtr>& subgraph_nodes) {
  std::vector<NodePtr> outNodes;
  for (auto n : graph.nodes_)
    if (!in_vec(subgraph_nodes, n))
      for (auto i : n->inputs_)
        if (in_vec(subgraph_nodes, i) && !in_vec(outNodes, i))
          outNodes.emplace_back(i);

  for (auto n : graph.outputs_)
    if (in_vec(subgraph_nodes, n)) outNodes.emplace_back(n);

  if (outNodes.size() == 0) {
    throw std::runtime_error(
        "This subgraph has no outputs, something is wrong!");
  }

  return outNodes;
}

// Find a subgraph, check it for bad branches
std::vector<NodePtr> FindSubgraph(const Graph& graph, NodePtr node,
                                  const std::function<bool(NodePtr)>& func) {
  // find simply connected nodes that are ngraph compatible
  auto subgraph_nodes = SelectNodes(node, func);

  // search for broken loops
  // remove nodes on broken loops
  auto outNodes = RemoveBroken(node, subgraph_nodes);

  return outNodes;
}

// function to identify and label connected ngraph ops as subgraphs
void IdentifySubgraphs(Graph* graph, const std::function<bool(NodePtr)>& func) {
  int sg = 1;
  bool found_subgraph = false;
  auto identify_inner_loop = [&graph, &func, &sg, &found_subgraph](NodePtr n) {
    if (n->subgraph_ == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto subgraph_nodes = FindSubgraph(*graph, n, func);

      // if we found a significantly large subgraph, label it
      if (subgraph_nodes.size() > 0) {
        for (auto node : subgraph_nodes) {
          node->subgraph_ = sg;
        }
        CollapseSubgraphs(graph, sg);
        found_subgraph = true;
        sg += 1;
      }
    }
  };

  // collapse graphs from the outputs
  for (size_t i = 0; i < graph->num_outputs_; ++i) {
    identify_inner_loop(graph->outputs_[i]);
  }
  // loop over the nodes from the back
  while (true) {
    found_subgraph = false;
    for (auto n = graph->nodes_.rbegin(); n != graph->nodes_.rend(); ++n) {
      identify_inner_loop(*n);
      if (found_subgraph) break;
    }
    if (found_subgraph) {
      continue;
    } else {
      break;
    }
  }
}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void CollapseSubgraphs(Graph* graph, int subgraph_num) {
  // loop variable for undefined number of subgraphs
  auto tmpGraph = std::make_shared<Graph>(
      "subgraph_" + randomString(12) + std::to_string(subgraph_num),
      graph->context_);
  // loop over all nodes and add nodes in the current subgraph to
  for (auto node : graph->nodes_) {
    if (node->subgraph_ == subgraph_num) {
      tmpGraph->AddNode(node);
    }
  }

  if (tmpGraph->nodes_.size() != 0) {
    tmpGraph->outputs_ = GetSubgraphOutputs(*graph, tmpGraph->nodes_);
    tmpGraph->num_outputs_ = tmpGraph->outputs_.size();
    for (size_t i = 0; i < tmpGraph->outputs_.size(); ++i) {
      tmpGraph->output_elements_.emplace_back(
          std::make_shared<OutputElement>(tmpGraph, i));
      tmpGraph->output_elements_.back()->subgraph_ = subgraph_num;
    }
    // if we found nodes, setup subgraph
    tmpGraph->in_ngraph_ = true;
    tmpGraph->subgraph_ = subgraph_num;

    auto in_tmpGraphInputs = [&tmpGraph](NodePtr n) {
      if (!in_vec(tmpGraph->inputs_, n)) return false;
      return true;
    };

    // setup inputs to this subgraph (as a node)
    for (auto node : tmpGraph->nodes_) {
      for (auto input : node->inputs_) {
        if (input->subgraph_ != i && !in_tmpGraphInputs(input))
          tmpGraph->inputs_.emplace_back(input);

    std::unordered_map<NodePtr, NodePtr> output_map;
    for (auto output : tmpGraph->output_elements_) {
      output_map.insert({output->base_node_, output});
    }

    // set any needed graph outputs to sub_graph outputs
    for (size_t i = 0; i < graph->outputs_.size(); ++i) {
      if (output_map.count(graph->outputs_[i])) {
        graph->outputs_[i] = output_map[graph->outputs_[i]];
      }
    }
    // insert the new outputs
    for (auto output : tmpGraph->output_elements_) {
      auto it = std::find_if(
          graph->nodes_.begin(), graph->nodes_.end(),
          [output](NodePtr n) -> bool { return (n == output->base_node_); });
      graph->nodes_.insert(it, output);
    }
    // delete all the nodes we're replacing with the subgraph
    graph->nodes_.erase(
        std::remove_if(graph->nodes_.begin(), graph->nodes_.end(),
                       [](NodePtr n) -> bool {
                         return ((n->subgraph_ > 0) &&
                                 (n->type_ == NodeType::kOp));
                       }),
        graph->nodes_.end());

    // set subgraph as input to all of the nodes.
    for (auto n : graph->nodes_) {
      for (size_t i = 0; i < n->inputs_.size(); ++i) {
        if (output_map.count(n->inputs_[i])) {
          n->inputs_[i] = output_map[n->inputs_[i]];
        }
      }
    }
    // set new subgraph as inputs to other subgraphs
    for (auto n : graph->nodes_) {
      if (n->type_ == NodeType::kGraph) {
        for (auto node : std::dynamic_pointer_cast<Graph>(n)->nodes_) {
          for (size_t i = 0; i < node->inputs_.size(); ++i) {
            if (output_map.count(node->inputs_[i])) {
              node->inputs_[i] = output_map[node->inputs_[i]];
            }
          }
        }
      }
    }
    // add the subraph to to Graph nodes
    graph->nodes_.push_back(tmpGraph);
  }
}

}  // namespace ngraph_bridge
