#include "ngraph_compiler.h"


namespace ngraph{
    UnaryOps PyCompiler::create_UnaryOps(const py::module& ns, const py::module& ng){
        UnaryOps output;
        for (auto op : {"abs", "exp", "tanh", "sigmoid", "relu", 
                        "log", "negative", "square", "sign",} )
            output[op] = [ns, ng, op](const py::object& py_operand){
                return ns.attr(op)(py_operand).attr("named")(op);
            };
        return output;
    }

    BinaryOps PyCompiler::create_BinaryOps(const py::module& ns, const py::module& ng){
        BinaryOps output;
        for (auto op : { "add", "divide", "equal", "greater_equal", "greater",
                         "less_equal", "less", "maximum", "minimum", "multiply", 
                         "not_equal", "pow", "mod", "subtract", "matmul"} )
            output[op] = [ns, ng, op](const py::object& lhs, 
                                         const py::object& rhs){
                return ns.attr(op)(lhs, rhs).attr("named")(op);
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

      NgraphUnaryOps_ = create_UnaryOps(ns_, ng_);
      NgraphBinaryOps_ = create_BinaryOps(ns_, ng_);

      for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
      for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
      

    }

    void PyCompiler::CheckGraph(Graph graph){
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
        CheckGraph(g);
        g.WriteDot("test.dot");


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
        placeholder_order[placeholder_order.size()]=node->name;
    }

} //end namespace ngraph
