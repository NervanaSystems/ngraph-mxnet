#ifndef NGRAPH_INTERMEDIARY_GRAPH_H_
#define NGRAPH_INTERMEDIARY_GRAPH_H_


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>

namespace ngraph {

enum class NodeType{
    kData,
    kOp,
    kVariable
};

struct Node;

using NodePtr = std::shared_ptr<Node>;

struct Node {
    Node(NodeType t, std::string s):  type(t), name(s){};
    Node(NodeType t, std::string s, std::string op): 
        type(t), name(s), operation(op){};
    Node(NodeType t, std::string s, std::string op, 
         std::unordered_map<std::string, std::string> d): 
        type(t), name(s), operation(op), dict(d){};
    NodeType type;
    std::string name;
    std::string operation;
    std::vector<NodePtr> inputs;
    std::unordered_map<std::string, std::string> dict;
};

class Graph{
public:
    void AddNode(NodePtr node){nodes_.emplace_back(node);};
    void WriteDot(std::string fname);
    NodePtr operator[](std::string name){
        for (auto n : nodes_)
            if (n->name == name)
                return n;
        throw "node not in graph";
    };
    std::vector<NodePtr> nodes_;
};

Graph ParseNNVMSymbol(nnvm::Symbol& symbol);

} //end namespace ngraph

#endif