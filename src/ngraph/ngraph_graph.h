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

namespace ngraph_bridge {

// Forward Delcaration for type aliases
class Node;

// Useful type aliases
using nnvmNodePtr = std::shared_ptr<nnvm::Node>;
using NodePtr = std::shared_ptr<Node>;

// Possible Types of nodes in Current Version
enum class NodeType {kVariable, kAux, kOp, kGraph};

// Base class for Nodes in Intermediary Analysis Graph
class Node {
 public:
  Node(NodeType t, const nnvmNodePtr n, std::string s)
      : type_(t), orig_node_(n), name_(s){};
  Node(NodeType t, const nnvmNodePtr n, std::string s, std::vector<NodePtr> i)
      : type_(t), orig_node_(n), name_(s), inputs_(i){};

  // Function to create node label, used to export graph to graphviz for debug
  virtual std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape_ << " sg=" << subgraph_;
    return name_ + " [label = \"" + name_ + "\n" + stream.str() +
           "\", fillcolor = red, style = filled];";
  }
  // basic information about node
  NodeType type_;
  nnvmNodePtr orig_node_;
  std::string name_;
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

// Variable Node
class VariableNode : public Node {
 public:
  VariableNode(const nnvmNodePtr n, std::string s)
      : Node(NodeType::kVariable, n, s){};
  VariableNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i)
      : Node(NodeType::kVariable, n, s, i){};
};

// Aux Node
class AuxNode : public Node {
 public:
  AuxNode(const nnvmNodePtr n, std::string s)
      : Node(NodeType::kAux, n, s){};
  AuxNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i)
      : Node(NodeType::kAux, n, s, i){};
};

// Operation node
class OpNode : public Node {
 public:
  // Include operation in graphviz
  std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape_ << " sg=" << subgraph_;
    std::string out =
        name_ + " [label=\"" + name_ + "\nOp: " + operation_ + stream.str() + "\"";
    if (in_ngraph_) out += ", fillcolor = red, style = filled";
    out += "];";
    return out;
  }
  OpNode(const nnvmNodePtr n, std::string s, std::string o)
      : Node(NodeType::kOp, n, s) {
    operation_ = o;
  };
  OpNode(const nnvmNodePtr n, std::string s, std::string o,
         std::vector<NodePtr> i)
      : Node(NodeType::kOp, n, s, i) {
    operation_ = o;
  };
};

using edgeRemoveTup = std::tuple<NodePtr, NodePtr, bool>;

/*
Graph class
Graph subclasses Node so that we can embed graphs into other graphs
This is useful when we take a graph and replace it with an ngraph computation
*/
class Graph : public Node {
 public:
  Graph() : Node(NodeType::kGraph, nullptr, ""){};
  Graph(std::string name) : Node(NodeType::kGraph, nullptr, name){};
  // Add a node to the graph
  void AddNode(NodePtr node) { nodes_.emplace_back(node); };
  // Write the graph to a Graphviz file
  void WriteDot(const std::string& fname);
  // Function for doing depth first search on the graph and selecting nodes
  std::vector<NodePtr> DFSselect(NodePtr s, std::function<bool(NodePtr)> func);
  // Utility function for graph search
  void DFSUtil(NodePtr s, std::unordered_set<NodePtr>& visited,
               std::vector<NodePtr>& outNodes,
               std::function<bool(NodePtr)>& func);
  
  // void findBrokenLoop(std::function<bool(NodePtr)> func);
  void IdentifySubgraphs(std::function<bool(NodePtr)> func);
  std::vector<NodePtr> FindSubgraph(NodePtr s,
                                    std::function<bool(NodePtr)> func);
  std::vector<NodePtr> RemoveBroken(NodePtr s,
                                    std::vector<NodePtr>& subgraph_nodes,
                                    std::function<bool(NodePtr)> func);
  void RemoveUtil(NodePtr s, std::vector<NodePtr>& outNodes,
                  std::function<bool(NodePtr)> func, 
                  std::set<edgeRemoveTup>& visited_edges);
  std::vector<NodePtr> PruneSubgraphOutputs(
      NodePtr s, std::vector<NodePtr>& subgraph_nodes,
      std::function<bool(NodePtr)> func);
  // convert graph from identified nodes to a network of nodes and graphs,
  // each graph node represented a combined ngraph operation
  void CollapseSubgraphs();

  // get the node corresponding to a name
  NodePtr operator[](std::string name) {
    for (auto n : nodes_)
      if (n->name_ == name) return n;
    throw "NGRAPH_BRIDGE: node not in graph";
  };

  void WriteSubgraphDots(std::string base){
    WriteDot(base + ".dot");
    for (auto n : nodes_) {
      if (n->type_ == NodeType::kGraph) {
        auto sg = std::dynamic_pointer_cast<Graph>(n);
        std::ostringstream stream;
        stream << base << sg->subgraph_ << ".dot";
        sg->WriteDot(stream.str());
      }
    }
  }

  int num_outputs = 1;
  std::vector<NodePtr> nodes_;
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_forward;
  std::shared_ptr<ngraph::runtime::CallFrame> ngraph_backward;
};


}  // end namespace ngraph

#endif
