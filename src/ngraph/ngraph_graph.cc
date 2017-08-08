#include "ngraph_graph.h"
using namespace pybind11::literals;
#include <map>
#include <functional>
#include <algorithm>


namespace ngraph{

    using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)> >;

    static layerGraphs create_layerGraphs(){
        layerGraphs layer_funcs;

        layer_funcs[std::string("FullyConnected")] = [](const NodePtr node){
            Graph tmpGraph;
            auto dotop = std::make_shared<OpNode>(node->orig_node, "dot_" + node->name, "matmul");
            dotop->inputs.emplace_back(node->inputs[1]);
            dotop->inputs.emplace_back(node->inputs[0]);
            tmpGraph.AddNode(dotop);
            auto addop = std::make_shared<OpNode>(node->orig_node, node->name, "add");
            addop->inputs.emplace_back(dotop);
            addop->inputs.emplace_back(node->inputs[2]);
            tmpGraph.AddNode(addop);
            return tmpGraph;
        };

        layer_funcs[std::string("Activation")] = [](const NodePtr node){
            Graph tmpGraph;
            auto act_type = node->orig_node->attrs.dict["act_type"];
            auto inputs = node->inputs;
            if (act_type == "tanh" || 
                act_type == "sigmoid" || 
                act_type == "relu"){
                tmpGraph.AddNode(std::make_shared<OpNode>(
                    node->orig_node, node->name, 
                    act_type, node->inputs));
            } else if (act_type == "softrelu"){
                auto one = std::make_shared<VariableNode>(node->orig_node, "ones_like_"+node->name);
                tmpGraph.AddNode(one);
                auto exp = std::make_shared<OpNode>(node->orig_node, node->name + "_exp", "exp");
                exp->inputs = inputs;
                tmpGraph.AddNode(exp);
                auto add = std::make_shared<OpNode>(node->orig_node, node->name + "_add", "add");
                add->inputs.emplace_back(one);
                add->inputs.emplace_back(exp);
                tmpGraph.AddNode(add);
                auto log = std::make_shared<OpNode>(node->orig_node, node->name, "log");
                log->inputs.emplace_back(add);
                tmpGraph.AddNode(add);
            }
            
            return tmpGraph;
        };
        return layer_funcs;
    }

    auto layer_funcs = create_layerGraphs();
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


    void Graph::WriteDot(const std::string& fname){
        std::ofstream dotfile;
        dotfile.open(fname);
        dotfile << "digraph G { " << std::endl;
        dotfile << "size=\"8,10.5\"" <<std::endl;
        for (auto n : nodes_){
            for (auto i : n->inputs){
                dotfile << i->name << " -> " << n->name << ";" <<std::endl;
            }
        }
        for (auto n : nodes_){
            dotfile << n->createNodeLabel() << std::endl ;
        }
        dotfile << "}" << std::endl;
        dotfile.close();
    }

    Graph ParseNNVMGraph(nnvm::Graph& graph, const size_t num_forward_inputs){
        Graph tmpGraph;
        nnvm::DFSVisit(graph.outputs, 
            [&tmpGraph](const nnvm::NodePtr node) {

                if (node->is_variable()) {
                  tmpGraph.AddNode(std::make_shared<VariableNode>(node, node->attrs.name));
                } else {
                  auto op_node = std::make_shared<OpNode>(node, node->attrs.name, node->op()->name) ;

                  for (size_t i = 0; i < node->inputs.size(); ++i) {
                    const nnvm::NodeEntry& e = node->inputs[i];
                    std::shared_ptr<Node> tmpnode;
                    try{
                        tmpnode = tmpGraph[e.node->attrs.name];
                    } catch (std::string & error){
                        tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
                        tmpGraph.AddNode(tmpnode);
                    }
                    op_node->inputs.emplace_back(tmpnode);
                  }
                  auto replace_subgraph = [&tmpGraph](OpNodePtr subgraph){
                        auto tmp = layer_funcs[subgraph->operation](subgraph);
                        for (auto n : tmp.nodes_){
                            tmpGraph.AddNode(n);
                        }                    
                  };
                  if (op_node->operation == "FullyConnected" ||
                      op_node->operation == "Activation"){ 
                      replace_subgraph(op_node);
                  } else {
                      tmpGraph.AddNode(op_node);
                  }

                  
                }
            }
        );

        // const auto& idx = graph.indexed_graph();
        // const auto inferred_shapes = graph.GetAttr<std::vector<nnvm::TShape>>("shape");
        // for (auto node : tmpGraph){
        //     const uint32_t nid = idx.node_id(node->orig_node);
        //     const uint32_t eid = idx.entry_id(nid, 0);
        //     const auto inferred_shape = inferred_shapes[eid];
        //     tmpGraph
        // }

        const auto& idx = graph.indexed_graph();
        const auto inferred_shapes = graph.GetAttr<std::vector<nnvm::TShape>>("shape");
        for (size_t i = 0; i < num_forward_inputs; ++i) {
            const uint32_t nid = idx.input_nodes().at(i);
            const uint32_t eid = idx.entry_id(nid, 0);
            const auto inferred_shape = inferred_shapes[eid];
            const std::string& arg_name = idx[nid].source->attrs.name;
            tmpGraph[arg_name]->shape = inferred_shape;
        }
        return tmpGraph;
    }


}