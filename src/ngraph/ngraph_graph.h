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

#ifndef NGRAPH_INTERMEDIARY_GRAPH_H_
#define NGRAPH_INTERMEDIARY_GRAPH_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <mxnet/base.h>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include <ngraph/ngraph.hpp>
#include "ngraph_graph_utils.h"

namespace ngraph_bridge {

// Useful type aliases
using NgraphNodePtr = std::shared_ptr<ngraph::Node>;
using nnvmNodePtr = std::shared_ptr<nnvm::Node>;

// Possible Types of nodes in Current Version
enum class NodeType { kVariable, kAux, kOp, kGraph };

// Base class for Nodes in Intermediary Analysis Graph
class Node {
 protected:
  Node(NodeType type, nnvmNodePtr node, const std::string& name)
      : type_(type),
        orig_node_(node),
        name_(name == "" ? randomString(6) : name) {}
  Node(NodeType type, nnvmNodePtr node, const std::string& name,
       const std::vector<NodePtr>& inputs)
      : type_(type),
        orig_node_(node),
        name_(name == "" ? randomString(6) : name),
        inputs_(inputs) {}

 public:
  // Function to create node label, used to export graph to graphviz for debug
  virtual std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape_ << " sg=" << subgraph_;
    return name_ + " [label = \"" + name_ + "\n" + stream.str() +
           "\", fillcolor = red, style = filled];";
  }
  // basic information about node
  const NodeType type_;
  const nnvmNodePtr orig_node_;
  const std::string name_;
  std::vector<NodePtr> inputs_;

  // mxnet type information
  nnvm::TShape shape_;
  int dtype_ = 0;

  // information to store graph parsing in
  int multi_output_index_ = -1;
  bool in_ngraph_ = false;
  std::string operation_ = "";
  int subgraph_ = 0;
};

// Class to store Variables
// Effectivly just a wrapper for setting Node Type
class VariableNode : public Node {
 public:
  // Overloaded constructors for ease of use
  VariableNode(nnvmNodePtr node, const std::string& name)
      : Node(NodeType::kVariable, node, name) {}
  VariableNode(nnvmNodePtr node, const std::string& name,
               const std::vector<NodePtr>& inputs)
      : Node(NodeType::kVariable, node, name, inputs) {}
};

// Class to store Auxillary Tensors, mostly for Batch Norm
// Effectivly just a wrapper for setting Node Type
class AuxNode : public Node {
 public:
  // Overloaded constructors for ease of use
  AuxNode(nnvmNodePtr node, const std::string& name)
      : Node(NodeType::kAux, node, name) {}
  AuxNode(nnvmNodePtr node, const std::string& name,
          const std::vector<NodePtr>& inputs)
      : Node(NodeType::kAux, node, name, inputs) {}
};

// Node for storing operations
class OpNode : public Node {
 public:
  // Include operation in graphviz
  std::string createNodeLabel() override {
    std::ostringstream stream;
    stream << shape_ << " sg=" << subgraph_;
    std::string out = name_ + " [label=\"" + name_ + "\nOp: " + operation_ +
                      stream.str() + "\"";
    if (in_ngraph_) out += ", fillcolor = red, style = filled";
    out += "];";
    return out;
  }

  // Overloaded constructors for ease of use
  OpNode(nnvmNodePtr node, const std::string& name,
         const std::string& operation)
      : Node(NodeType::kOp, node, name) {
    operation_ = operation;
  }
  OpNode(nnvmNodePtr node, const std::string& name,
         const std::string& operation, const std::vector<NodePtr>& inputs)
      : Node(NodeType::kOp, node, name, inputs) {
    operation_ = operation;
  }
};

inline std::shared_ptr<ngraph::runtime::Manager> GetManagerFromContext(
    const mxnet::Context& context) {
  if (context == mxnet::Context::NNP(0)) {
    return ngraph::runtime::Manager::get("ARGON");
  }

  return ngraph::runtime::Manager::get("NGVM");
}

struct FpropCache {
  ngraph::NodeMap nodes_to_params;
  ngraph::Nodes output_nodes;
  std::vector<std::shared_ptr<ngraph::op::Parameter>> input_params;
  std::vector<std::shared_ptr<ngraph::runtime::Value>> values;
};

/*
Graph class
Graph subclasses Node so that we can embed graphs into other graphs
This is useful when we take a graph and replace it with an ngraph computation
TODO: Refactor into Graph and subgraph?
*/
class Graph : public Node {
 public:
  Graph(const std::string& name = "",
        const mxnet::Context& context = mxnet::Context::CPU())
      : Node(NodeType::kGraph, nullptr, name),
        context_(context),
        manager_(GetManagerFromContext(context_)),
        backend_(manager_->allocate_backend()) {}

  // Add a node to the graph
  void AddNode(NodePtr node) { nodes_.emplace_back(node); }

  // get the node corresponding to a name
  NodePtr operator[](std::string name) {
    for (auto n : nodes_)
      if (n->name_ == name) return n;
    // This throw is used in constructing multi-output subgraphs
    throw "NGRAPH_BRIDGE: node not in graph";
  }

  int num_outputs = 1;
  // nodes in this graph
  std::vector<NodePtr> nodes_;
  // functions to execute this graph in ngraph
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_forward;
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_backward;

  const mxnet::Context context_;
  const std::shared_ptr<ngraph::runtime::Manager> manager_;
  const std::shared_ptr<ngraph::runtime::Backend> backend_;
  FpropCache fprop_cache;
};

/**
 * High level function that does the subgraph identification
 */
void IdentifySubgraphs(Graph& graph, std::function<bool(NodePtr)> func);

/**
 * Convert graph from identified nodes to a network of nodes and graphs,
 * each graph node represented a combined ngraph operation
 */
void CollapseSubgraphs(Graph& graph);

/**
 * Selection of nodes based on function criterion.
 * Note: uses DFSUtil().
 */
std::vector<NodePtr> SelectNodes(NodePtr node,
                                 std::function<bool(NodePtr)> func);

/**
 * Finds simply connected ngraph operations
 */
std::vector<NodePtr> FindSubgraph(Graph& graph, NodePtr node,
                                  std::function<bool(NodePtr)> func);

}  // namespace ngraph_bridge

#endif
