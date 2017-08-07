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

    class Node;

    using nnvmNodePtr = std::shared_ptr<nnvm::Node>;
    using NodePtr = std::shared_ptr<Node>;

    class Node {
    public:
        Node(const nnvmNodePtr n, std::string s): orig_node(n), name(s) {};
        Node(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i): 
            orig_node(n), name(s), inputs(i) {};
        virtual void Check_InNgraph();
        virtual std::string createNodeLabel(){
            return name + " [fillcolor = red, style = filled];";
        }

        nnvmNodePtr orig_node;
        std::string name;
        std::vector<NodePtr> inputs;
        bool in_ngraph = false;
    };

    class DataNode : public Node {
    public:
        DataNode(const nnvmNodePtr n, std::string s): Node(n,s) {};
        DataNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):  
            Node(n,s,i) {};
    };

    class VariableNode : public Node {
    public:
        VariableNode(const nnvmNodePtr n, std::string s):  Node(n,s) {};
        VariableNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):  
            Node(n,s,i) {};
    };

    class OpNode : public Node {
    public:
        std::string createNodeLabel(){
            std::string out = name + " [label=\"" + name 
                            + "\nOp: " + operation + "\"" ;
            if (in_ngraph) out += ", fillcolor = red, style = filled";
            out += "];";
            return out;
        }
        OpNode(const nnvmNodePtr n, std::string s, std::string o):
            Node(n,s), operation(o) {};
        OpNode(const nnvmNodePtr n, std::string s,
               std::string o, std::vector<NodePtr> i):  
            Node(n,s,i) , operation(o) {};

        void Check_InNgraph();
        std::string operation;
    };
    using OpNodePtr = std::shared_ptr<OpNode>;
    class Graph{
    public:

        void AddNode(NodePtr node){nodes_.emplace_back(node);};
        void WriteDot(const std::string& fname);
        NodePtr operator[](std::string name){
            for (auto n : nodes_)
                if (n->name == name)
                    return n;
            throw "node not in graph";
        };
        void Check_InNgraph(){for (auto n : nodes_) n->Check_InNgraph();};
        std::vector<NodePtr> nodes_;
    };

    class subGraph : public Graph{
        subGraph(Graph g);
    };

    Graph ParseNNVMGraph(const nnvm::Graph& graph);

} //end namespace ngraph

#endif