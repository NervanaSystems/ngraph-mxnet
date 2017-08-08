#include "ngraph_compiler.h"
#include "reverse_iterate.h"


namespace ngraph{
    UnaryOps PyCompiler::create_UnaryOps(const py::module& ns, const py::module& ng){
        UnaryOps output;
        for (auto op : {"abs", "exp", "tanh", "sigmoid", "relu", 
                        "log", "negative", "square", "sign",} )
            output[op] = [ns, op](const py::object& py_operand, 
                                  const std::string& name){
                return ns.attr(op)(py_operand).attr("named")(name);
            };
        for (auto op : {"flatten",} )
            output[op] = [ng, op](const py::object& py_operand, 
                                      const std::string& name){
                return ng.attr(op)(py_operand).attr("named")(name);
            };
        return output;
    }

    BinaryOps PyCompiler::create_BinaryOps(const py::module& ns, const py::module& ng){
        BinaryOps output;
        for (auto op : { "add", "divide", "equal", "greater_equal", "greater",
                         "less_equal", "less", "maximum", "minimum", "multiply", 
                         "not_equal", "pow", "mod", "subtract", "matmul"} )
            output[op] = [ns, op](const py::object& lhs, 
                                  const py::object& rhs,
                                  const std::string& name){
                return ns.attr(op)(lhs, rhs).attr("named")(name);
            };
        return output;
    }

    PyCompiler::PyCompiler() {
        // init python
        InitializePython();
        gil_state state;

        // // import python modules
        ng_ = py::module::import("ngraph");
        ns_ = py::module::import(
            "ngraph.frontends.tensorflow.tf_importer.ngraph_shaped");
        ngt_ = py::module::import("ngraph.transformers");
        transformer_ = ngt_.attr("make_transformer")();

        //Create Operation Maps
        NgraphUnaryOps_ = create_UnaryOps(ns_, ng_);
        NgraphBinaryOps_ = create_BinaryOps(ns_, ng_);

        // Find all the valid operation names
        for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
        for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
    }

    void PyCompiler::CheckInNGraph(Graph graph){
        for (auto node : graph.nodes_){
            if (node->type == NodeType::kOp){
                for (auto op : NgraphOps_){
                    if (node->operation == op){
                        node->in_ngraph = true;
                        break;
                    }
                }
            } else {
                node->in_ngraph=true;
            }
        }
    }

    void PyCompiler::Compile(nnvm::Graph& graph, const size_t num_forward_inputs){
        gil_state state;
        auto g = ParseNNVMGraph(graph, num_forward_inputs);
        CheckInNGraph(g);

        IdentifySubgraphs(g);

        CollapseSubgraphs(g);

        if (true){
            g.WriteDot("test.dot");
            for (auto n : g.nodes_){
                if (n->type == NodeType::kGraph){
                    auto sg = std::dynamic_pointer_cast<Graph>(n);
                    std::ostringstream stream;
                    stream << "test" << sg->subgraph << ".dot";
                    sg->WriteDot(stream.str());
                }
            }
        }
    }

    void PyCompiler::CollapseSubgraphs(Graph& graph){
        int i = 1;
        while (true) {
            auto tmpGraph = std::make_shared<Graph>();
            for (auto node : graph.nodes_)
                if (node->subgraph == i)
                    tmpGraph->AddNode(node);

            if (tmpGraph->nodes_.size()>0){
                tmpGraph->in_ngraph=true;
                tmpGraph->subgraph = i;
                auto name = tmpGraph->nodes_.back()->name;
                tmpGraph->name = "subgraph_" + name;
                for (auto node: tmpGraph->nodes_){
                    for (auto input: node->inputs){
                        if (input->subgraph != i)
                            tmpGraph->inputs.emplace_back(input);
                    }
                }
                auto it = std::find_if(graph.nodes_.begin(), graph.nodes_.end(), 
                        [name](NodePtr n) -> bool {return (n->name == name);});

                graph.nodes_.insert(it, tmpGraph);
                graph.nodes_.erase(std::remove_if(
                               graph.nodes_.begin(), 
                               graph.nodes_.end(),
                               [i](NodePtr n) -> bool {
                                    return ((n->subgraph == i) &&
                                            (n->type == NodeType::kOp));}), 
                               graph.nodes_.end());

                for (auto n : graph.nodes_)
                    for (size_t i=0; i < n->inputs.size(); ++i)
                        if (n->inputs[i]->name == name)
                            n->inputs[i] = tmpGraph;

            } else {
                break;
            }
            i += 1;
        }
    }

    void PyCompiler::IdentifySubgraphs(Graph& graph){
        int sg = 1;
        for (auto i : reverse_iterate(graph.nodes_)) {
            if (i->subgraph == 0){
                auto subgraph_nodes = graph.DFSselect(i,
                        [](NodePtr s){return s->in_ngraph;});
                if (subgraph_nodes.size()>2){
                    for (auto node : subgraph_nodes)
                        if (node->type == NodeType::kOp)
                            node->subgraph = sg;
                    sg += 1;
                }
            }
        } 
    }

    py::tuple TShapeToTuple(nnvm::TShape shape){
        py::tuple shape_tuple = py::make_tuple();
        for (auto x : shape){
            shape_tuple = shape_tuple.attr("__add__")(py::make_tuple(x));
        }
        return shape_tuple;
    }

    void PyCompiler::createPyPlaceholder(NodePtr node){
        using namespace pybind11::literals;
        py::tuple py_shape = TShapeToTuple(node->shape);
        op_map[node->name] = ns_.attr("placeholder")("shape"_a = py_shape
                               ).attr("named")(node->name);
        placeholder_order.emplace_back(node->name);
    }

} //end namespace ngraph
