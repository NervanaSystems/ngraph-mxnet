#include "ngraph_graph.h"
#include <map>
#include <functional>


namespace ngraph{

    const static std::vector<std::string> ngraph_ops({
            "add",
            "divide",
            "equal",
            "greater_equal",
            "greater",
            "less_equal",
            "less",
            "maximum",
            "minimum",
            "multiply",
            "not_equal",
            "pow",
            "mod",
            "subtract",
            "abs",
            "exp",
            "tanh",
            "sigmoid",
            "relu",
            "log",
            "negative",
            "square",
            "sign",
            "reduce_max",
            "reduce_mean",
            "reduce_min",
            "reduce_prod",
            "reduce_sum",
            "matmul",
    });

    void Node::Check_InNgraph(){
        if (type == NodeType::kVariable || type == NodeType::kData){
            in_ngraph = true;
        } else {
            for (auto op : ngraph_ops){
                if (op == operation){
                    in_ngraph = true;
                    break;
                }
            }
        }
    }

    typedef std::map<std::string, std::function<Graph(const NodePtr)> > layerGraphs;

    static layerGraphs create_layerGraphs(){
        layerGraphs layer_funcs;

        layer_funcs[std::string("FullyConnected")] = [](const NodePtr node){
            Graph tmpGraph;
            auto dotop = std::make_shared<Node>(NodeType::kOp, "dot_" + node->name, "matmul", node->dict);
            dotop->inputs.emplace_back(node->inputs[1]);
            dotop->inputs.emplace_back(node->inputs[0]);
            tmpGraph.AddNode(dotop);
            auto addop = std::make_shared<Node>(NodeType::kOp, node->name, "add", node->dict);
            addop->inputs.emplace_back(dotop);
            addop->inputs.emplace_back(node->inputs[2]);
            tmpGraph.AddNode(addop);
            return tmpGraph;
        };

        layer_funcs[std::string("Activation")] = [](const NodePtr node){
            Graph tmpGraph;
            auto act_type = node->dict["act_type"];
            auto inputs = node->inputs;
            if (act_type == "tanh" || 
                act_type == "sigmoid" || 
                act_type == "relu"){
                tmpGraph.AddNode(std::make_shared<Node>(
                    NodeType::kOp, node->name, 
                    act_type, node->dict, node->inputs));
            } else if (act_type == "softrelu"){
                auto one = std::make_shared<Node>(NodeType::kVariable, "ones_like_"+node->name);
                tmpGraph.AddNode(one);
                auto exp = std::make_shared<Node>(NodeType::kOp, node->name + "_exp", "exp");
                exp->inputs = inputs;
                tmpGraph.AddNode(exp);
                auto add = std::make_shared<Node>(NodeType::kOp, node->name + "_add", "add");
                add->inputs.emplace_back(one);
                add->inputs.emplace_back(exp);
                tmpGraph.AddNode(add);
                auto log = std::make_shared<Node>(NodeType::kOp, node->name, "log");
                log->inputs.emplace_back(add);
                tmpGraph.AddNode(add);
            }
            
            return tmpGraph;
        };
        return layer_funcs;
    }

    layerGraphs layer_funcs = create_layerGraphs();

    std::string createNodeLabel(NodePtr n){
        std::string nlabel;
        if (n->type == NodeType::kOp){
            nlabel = "Op: " + n->operation +"\n" +n->name;
        } else {
            nlabel = n->name;
        }
        return nlabel;
    }

    void Graph::WriteDot(std::string fname){
        std::ofstream dotfile;
        dotfile.open(fname);
        dotfile << "digraph G { " << std::endl;
        dotfile << "size=\"8,10.5\"" <<std::endl;
        Check_InNgraph();
        for (auto n : nodes_){
            for (auto i : n->inputs){
                dotfile << i->name << " -> " << n->name << ";" <<std::endl;
            }
        }
        for (auto n : nodes_){
            if (n->type == NodeType::kOp){
                dotfile << n->name << " [label=\"" << n->name + "\nOp: " + n->operation << "\"" ;
                if (n->in_ngraph)
                    dotfile << ", fillcolor = red, style = filled";
                dotfile << "];" << std::endl ;
            } else {
                dotfile << n->name << " [fillcolor = red, style = filled];" << std::endl ;
            }
        }
        dotfile << "}" << std::endl;
        dotfile.close();
    }


    Graph ParseNNVMSymbol(nnvm::Symbol& symbol){
        Graph tmpGraph;
        nnvm::DFSVisit(symbol.outputs, 
            [&tmpGraph](const nnvm::NodePtr node) {

                if (node->is_variable()) {
                  tmpGraph.AddNode(std::make_shared<Node>(NodeType::kVariable, node->attrs.name));
                } else {
                  auto op_node = std::make_shared<Node>(NodeType::kOp, node->attrs.name, node->op()->name, node->attrs.dict) ;

                  for (size_t i = 0; i < node->inputs.size(); ++i) {
                    const nnvm::NodeEntry& e = node->inputs[i];
                    std::shared_ptr<Node> tmpnode;
                    try{
                        tmpnode = tmpGraph[e.node->attrs.name];
                    } catch (std::string & error){
                        tmpnode = std::make_shared<Node>(NodeType::kVariable, e.node->attrs.name);
                        tmpGraph.AddNode(tmpnode);
                    }
                    op_node->inputs.emplace_back(tmpnode);
                  }
                  auto replace_subgraph = [&tmpGraph](NodePtr subgraph){
                        auto tmp = layer_funcs[subgraph->operation](subgraph);
                        for (auto n : tmp.nodes_){
                            tmpGraph.AddNode(n);
                        }                    
                  };
                  if (op_node->operation == std::string("FullyConnected") || 
                      op_node->operation == std::string("Activation")){ 
                      replace_subgraph(op_node);
                  } else {
                      tmpGraph.AddNode(op_node);
                  }
                  
                }
            }
        );
        tmpGraph.Check_InNgraph();
        return tmpGraph;
    }


}