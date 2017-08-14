#ifndef NGRAPH_INTERMEDIARY_GRAPH_H_
#define NGRAPH_INTERMEDIARY_GRAPH_H_


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include "ngraph_utils.h"

namespace ngraph {

class Node;

using nnvmNodePtr = std::shared_ptr<nnvm::Node>;
using NodePtr = std::shared_ptr<Node>;

enum class NodeType {kVariable, kOp, kGraph};
class Node {
 public:
  Node(NodeType t, const nnvmNodePtr n, std::string s): 
    type(t), orig_node(n), name(s) {};
  Node(NodeType t, const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):
    type(t), orig_node(n), name(s), inputs(i) {};

  virtual std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape << " sg=" << subgraph;
    return name + " [label = \"" + name + "\n" + stream.str() +
           "\", fillcolor = red, style = filled];";
  }

  NodeType type;
  nnvmNodePtr orig_node;
  std::string name;
  std::vector<NodePtr> inputs;
  nnvm::TShape shape;
  int dtype;
  bool in_ngraph = false;
  std::string operation = "";
  py::object ngraph_rep;
  int subgraph = 0;
};

class VariableNode : public Node {
 public:
  VariableNode(const nnvmNodePtr n, std::string s):
    Node(NodeType::kVariable, n, s) {};
  VariableNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):
    Node(NodeType::kVariable, n, s, i) {};
};

class OpNode : public Node {
 public:
  std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape << " sg=" << subgraph;
    std::string out = name + " [label=\"" + name
                      + "\nOp: " + operation + stream.str() + "\"";
    if (in_ngraph) out += ", fillcolor = red, style = filled";
    out += "];";
    return out;
  }
  OpNode(const nnvmNodePtr n, std::string s, std::string o):
    Node(NodeType::kOp, n, s) {operation = o;};
  OpNode(const nnvmNodePtr n, std::string s,
         std::string o, std::vector<NodePtr> i):
    Node(NodeType::kOp, n, s, i) {operation = o;};

};


class Graph : public Node {
 public:
  Graph(): Node(NodeType::kGraph, nullptr, "") {};
  Graph(std::string name): Node(NodeType::kGraph, nullptr, name) {};
  void AddNode(NodePtr node) {nodes_.emplace_back(node);};
  void WriteDot(const std::string& fname);
  std::vector<NodePtr> DFSselect(NodePtr s, std::function<bool(NodePtr)> func);
  void DFSUtil(NodePtr s,
               std::map<std::string, bool>& visited,
               std::vector<NodePtr>& outNodes,
               std::function<bool(NodePtr)>& func);
  NodePtr operator[](std::string name) {
    for (auto n : nodes_)
      if (n->name == name)
        return n;
    throw "node not in graph";
  };
  std::vector<NodePtr> nodes_;
  std::shared_ptr<py::object> py_computation;
};


Graph ParseNNVMGraph(nnvm::Graph& graphs);

} //end namespace ngraph

#endif