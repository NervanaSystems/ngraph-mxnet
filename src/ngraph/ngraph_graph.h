#ifndef NGRAPH_INTERMEDIARY_GRAPH_H_
#define NGRAPH_INTERMEDIARY_GRAPH_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include "ngraph_utils.h"

namespace ngraph {

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
      : type(t), orig_node(n), name(s){};
  Node(NodeType t, const nnvmNodePtr n, std::string s, std::vector<NodePtr> i)
      : type(t), orig_node(n), name(s), inputs(i){};

  // Function to create node label, used to export graph to graphviz for debug
  virtual std::string createNodeLabel() {
    std::ostringstream stream;
    stream << shape << " sg=" << subgraph;
    return name + " [label = \"" + name + "\n" + stream.str() +
           "\", fillcolor = red, style = filled];";
  }
  // basic information about node
  NodeType type;
  nnvmNodePtr orig_node;
  std::string name;
  std::vector<NodePtr> inputs;

  // mxnet type information
  nnvm::TShape shape;
  int dtype;
  // information to store graph parsing in
  bool in_ngraph = false;
  std::string operation = "";
  int subgraph = 0;
  py::object ngraph_rep;
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
    stream << shape << " sg=" << subgraph;
    std::string out =
        name + " [label=\"" + name + "\nOp: " + operation + stream.str() + "\"";
    if (in_ngraph) out += ", fillcolor = red, style = filled";
    out += "];";
    return out;
  }
  OpNode(const nnvmNodePtr n, std::string s, std::string o)
      : Node(NodeType::kOp, n, s) {
    operation = o;
  };
  OpNode(const nnvmNodePtr n, std::string s, std::string o,
         std::vector<NodePtr> i)
      : Node(NodeType::kOp, n, s, i) {
    operation = o;
  };
};

/*
Graph class
Graph subclasses Node to that we can embed graphs into other graphs
This is useful when we take a graph and replace it with an ngraph computation
*/
using edgeRemoveTup = std::tuple<std::string, std::string, bool>;

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
  void DFSUtil(NodePtr s, std::map<std::string, bool>& visited,
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
                  std::vector<edgeRemoveTup>& visited_edges);

  // convert graph from identified nodes to a network of nodes and graphs,
  // each graph node represented a combined ngraph operation
  void CollapseSubgraphs();

  // get the node corresponding to a name
  NodePtr operator[](std::string name) {
    for (auto n : nodes_)
      if (n->name == name) return n;
    throw "node not in graph";
  };

  void WriteSubgraphDots(std::string base){
    WriteDot(base + ".dot");
    for (auto n : nodes_) {
      if (n->type == NodeType::kGraph) {
        auto sg = std::dynamic_pointer_cast<Graph>(n);
        std::ostringstream stream;
        stream << base << sg->subgraph << ".dot";
        sg->WriteDot(stream.str());
      }
    }
  }

  std::vector<NodePtr> nodes_;
  std::shared_ptr<py::object> py_computation;
  std::shared_ptr<py::object> py_backward;
};


}  // end namespace ngraph

#endif