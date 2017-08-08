#ifndef NGRAPH_INTERMEDIARY_GRAPH_H_
#define NGRAPH_INTERMEDIARY_GRAPH_H_


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include "ngraph_utils.h"

namespace ngraph {

    class Node;

    using nnvmNodePtr = std::shared_ptr<nnvm::Node>;
    using NodePtr = std::shared_ptr<Node>;

    enum class NodeType {kData,kVariable,kOp};
    class Node {
    public:
        Node(NodeType t, const nnvmNodePtr n, std::string s): 
            type(t), orig_node(n), name(s) {};
        Node(NodeType t, const nnvmNodePtr n, std::string s, 
             std::vector<NodePtr> i): 
                type(t), orig_node(n), name(s), inputs(i) {};
        virtual std::string createNodeLabel(){
            std::ostringstream stream;
            stream << shape;
            return name + " [label = \"" + name + "\n" + stream.str() + 
                             "\", fillcolor = red, style = filled];";
        }
        // virtual py::object getPyNode(py::module& ng, py::module& ns);
        NodeType type;
        nnvmNodePtr orig_node;
        std::string name;
        std::vector<NodePtr> inputs;
        nnvm::TShape shape;
        bool in_ngraph = false;
        std::string operation = "";
    };

    class DataNode : public Node {
    public:
        DataNode(const nnvmNodePtr n, std::string s): 
            Node(NodeType::kData, n,s) {};
        DataNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):  
            Node(NodeType::kData, n,s,i) {};
    };

    class VariableNode : public Node {
    public:
        VariableNode(const nnvmNodePtr n, std::string s):  
            Node(NodeType::kVariable,n,s) {};
        VariableNode(const nnvmNodePtr n, std::string s, std::vector<NodePtr> i):  
            Node(NodeType::kVariable,n,s,i) {};
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
            Node(NodeType::kOp,n,s) {operation = o;};
        OpNode(const nnvmNodePtr n, std::string s,
               std::string o, std::vector<NodePtr> i):  
            Node(NodeType::kOp,n,s,i){operation = o;};

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
        std::vector<NodePtr> nodes_;
    };

    class subGraph : public Graph{
        subGraph(Graph g);
    };

    Graph ParseNNVMGraph(nnvm::Graph& graph, const size_t num_forward_inputs);

} //end namespace ngraph

#endif