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
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <unordered_set>
#include <set>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include <ngraph/ngraph.hpp>
#include "ngraph_graph_utils.h"

namespace ngraph_bridge {

// Forward Delcaration for type aliases
class Node;

// Useful type aliases
using nnvmNodePtr = std::shared_ptr<nnvm::Node>;
using NodePtr = std::shared_ptr<Node>;

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

using edgeRemoveTup = std::tuple<NodePtr, NodePtr, bool>;

/**
 * Graph class
 * Graph subclasses Node so that we can embed graphs into other graphs
 * This is useful when we take a graph and replace it with an ngraph computation
 * TODO: Refactor into Graph and subgraph?
 */
class Graph : public Node {
 public:
  Graph() : Graph("") {}
  Graph(const std::string &name) :
      Node(NodeType::kGraph, nullptr, name),
      num_outputs_(1) {}

  /**
   * Add a node to the graph
   * @param node
   */
  void AddNode(NodePtr node) { nodes_.emplace_back(node); }

  /**
   * Constant accessor for nodes
   * @return the nodes in the graph
   */
  const std::vector<NodePtr>& Graph::GetNodes() const { return nodes_; }

  /**
   * Non-const accessor for nodes
   * @return the nodes in the graph
   */
  std::vector<NodePtr>& Graph::GetNodes() { return nodes_; }

  /**
   * Accessor for number of outputs
   * @return number of outputs of this graph
   */
  int Graph::GetNumOutputs() const { return num_outputs_; }

  /**
   * Sets the number of outputs for this graph
   * @param num_outputs
   */
  void Graph::SetNumOutputs(int num_outputs) { num_outputs_ = num_outputs; }

  /**
   * High level function that does the subgraph identification
   * @param func
   */
  void IdentifySubgraphs(std::function<bool(NodePtr)> func);

  /**
   * convert graph from identified nodes to a network of nodes and graphs,
   * each graph node represented a combined ngraph operation
   */
  void CollapseSubgraphs();

  /**
   * get the node corresponding to a name
   * @param name
   * @return
   */
  NodePtr operator[](std::string name) {
    for (auto n : nodes_)
      if (n->name_ == name) return n;
    // This throw is used in constructing multi-output subgraphs
    throw "NGRAPH_BRIDGE: node not in graph";
  }

 private:
  int num_outputs_;
  // nodes in this graph
  std::vector<NodePtr> nodes_;
  // functions to execute this graph in ngraph
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_forward;
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_backward;
};

template <typename T>
inline bool in_vec(const std::vector<T>& vec, const T& s) {
  return (std::find(vec.begin(), vec.end(), s) != vec.end());
}

/**
 * Selection of nodes based on function criterion.
 * Note: uses DFSUtil().
 * @param s
 * @param func
 * @return
 */
std::vector<NodePtr> SelectNodes(NodePtr s, std::function<bool(NodePtr)> func);

/**
 * Utility to mark a node as visited and recursive search based on the results
 * of an input function
 * @param s
 * @param visited
 * @param outNodes
 * @param func
 */
void DFSUtil(NodePtr s,
             std::unordered_set<NodePtr>& visited,
             std::vector<NodePtr>& outNodes,
             std::function<bool(NodePtr)>& func);

/**
 * Graph pass find loops in the subgraph where 1 branch of the loop is ngraph
 * compatible and the other
 * @param s
 * @param subgraph_nodes
 * @param func
 * @return
 */
std::vector<NodePtr> RemoveBroken(NodePtr s,
                                  std::vector<NodePtr>& subgraph_nodes,
                                  std::function<bool(NodePtr)> func);

/**
 *
 * @param s
 * @param outNodes
 * @param func
 * @param visited_edges
 */
void RemoveUtil(NodePtr s,
                std::vector<NodePtr>& outNodes,
                std::function<bool(NodePtr)> func,
                std::set<edgeRemoveTup>& visited_edges);

/**
 * Modified subgraph to only return 1 output.
 * If we improve the subgraph compiler/nnvm op construction
 * we might be able to get rid of this pass
 * @param s
 * @param subgraph_nodes
 * @param func
 * @return
 */
std::vector<NodePtr> PruneSubgraphOutputs(NodePtr s,
                                          std::vector<NodePtr>& subgraph_nodes,
                                          std::function<bool(NodePtr)> func);



/**
 * Finds simply connected ngraph operations
 * @param s
 * @param func
 * @return
 */
std::vector<NodePtr> FindSubgraph(NodePtr s,
                                  std::function<bool(NodePtr)> func);

}  // namespace ngraph_bridge


#endif
