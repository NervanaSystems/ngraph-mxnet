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

std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Backend>>
    backends;

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

std::vector<NodePtr> DFSTopologicalSort(const std::vector<NodePtr>& outputs) {
  GraphVisitor visitor;

  std::unordered_set<NodePtr> visited;
  std::vector<NodePtr> sorted_nodes;

  visitor.operation = [&sorted_nodes, &visited](NodePtr node) {
    visited.insert(node);
    sorted_nodes.emplace_back(node);
  };
  visitor.stop_condition = [&visited](NodePtr node, NodePtr input) {
    if (!(visited.count(input))) {
      return false;
    }
    return true;
  };
  for (auto& node : outputs) {
    GraphTraverse(node, visitor);
  }
  return sorted_nodes;
}

std::vector<NodePtr> GetSubgraphOutputs(
    const Graph& graph, const std::vector<NodePtr>& subgraph_nodes) {
  std::unordered_set<NodePtr> sg_nodes(subgraph_nodes.begin(),
                                       subgraph_nodes.end());
  std::unordered_set<NodePtr> out_nodes_set;

  std::vector<NodePtr> outNodes;
  // of nodes in the subgraph are outputs of the main graph, they need
  // to be outputs of the subgraph
  for (auto n : graph.outputs_) {
    if (sg_nodes.count(n)) {
      outNodes.emplace_back(n);
    }
  }
  auto sorted_nodes = DFSTopologicalSort(graph.outputs_);

  // for every node in the subgraph, if the node is an input to other nodes
  // that aren't in the subgraph, this node is an output of the subgraph
  for (auto n : graph.nodes_) {
    if (!sg_nodes.count(n)) {
      for (auto i : n->inputs_) {
        if (sg_nodes.count(i) && !out_nodes_set.count(i)) {
          out_nodes_set.insert(i);
        }
      }
    }
  }

  // Add the outputs in reverse topological order
  for (auto it = sorted_nodes.rbegin(); it < sorted_nodes.rend(); ++it) {
    if (out_nodes_set.count(*it)) {
      outNodes.emplace_back(*it);
    }
  }

  if (outNodes.size() == 0) {
    // this is here for algorithm debugging
    throw std::runtime_error(
        "This subgraph has no outputs, something is wrong!");
  }

  return outNodes;
}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void CollapseSubgraph(Graph* graph, int subgraph_num) {
  std::unordered_set<NodePtr> orig_nodes(begin(graph->nodes_),
                                         end(graph->nodes_));
  // loop variable for undefined number of subgraphs
  auto tmpGraph = std::make_shared<Graph>(
      graph->name_ + "_subgraph_" + std::to_string(subgraph_num),
      graph->context_);

  // loop over all nodes and add nodes in the current subgraph to
  for (auto node : graph->nodes_) {
    if (node->subgraph_ == subgraph_num) {
      tmpGraph->AddNode(node);
    }
  }

  if (!tmpGraph->nodes_.empty()) {
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

    GraphVisitor visitor;

    std::unordered_set<NodePtr> nodes(tmpGraph->nodes_.begin(),
                                      tmpGraph->nodes_.end());
    std::unordered_set<NodePtr> visited;

    visitor.operation = [tmpGraph, &nodes, &visited](NodePtr node) {
      visited.insert(node);
      if (!nodes.count(node)) {
        tmpGraph->inputs_.push_back(node);
      }
    };
    visitor.stop_condition = [&nodes, &visited](NodePtr node, NodePtr input) {
      if (nodes.count(node) && !(visited.count(input))) {
        return false;
      }
      return true;
    };
    for (auto& node : tmpGraph->outputs_) {
      GraphTraverse(node, visitor);
    }

    for (auto input : tmpGraph->inputs_) {
      tmpGraph->input_is_weight_.push_back(false);
    }
    for (auto node : tmpGraph->nodes_) {
      if (node->operation_ == "Convolution" ||
          node->operation_ == "Deconvolution" ||
          node->operation_ == "BatchNorm" ||
          node->operation_ == "FullyConnected") {
        for (size_t i = 1; i < node->inputs_.size(); ++i) {
          auto input = node->inputs_[i];
          if (input->type_ == NodeType::kVariable ||
              input->type_ == NodeType::kAux) {
            tmpGraph
                ->input_is_weight_[std::find(tmpGraph->inputs_.begin(),
                                             tmpGraph->inputs_.end(), input) -
                                   tmpGraph->inputs_.begin()] = true;
          }
        }
      }
    }
    // create a map between base nodes and outputs for easier replacement
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
                       [subgraph_num, orig_nodes](NodePtr n) -> bool {
                         return (n->subgraph_ == subgraph_num &&
                                 orig_nodes.count(n));
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

void Node::printOpDetails(std::ostream& os) {
  using namespace std;
  // The set of fields printed can be altered according to the developer's
  // debugging needs.
  const string indent{"  "};

  os << "name_ = '" << name_ << "'" << endl;

  os << "orig_node_->attrs.dict:" << endl;
  for (const auto& kv : orig_node_->attrs.dict) {
    os << indent << kv.first << " = '" << kv.second << "'" << endl;
  }
}

}  // namespace ngraph_bridge
