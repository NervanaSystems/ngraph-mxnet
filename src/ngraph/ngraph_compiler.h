#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_utils.h"
#include "ngraph_graph.h"
#include "ngraph_emitter.h"

namespace ngraph {
using UnaryOps = std::map<std::string, std::function<py::object(const py::object&, const std::string&)> >;
using BinaryOps = std::map<std::string, std::function<py::object(const py::object&, const py::object&, const std::string&)> >;

class PyCompiler {
  public:
    PyCompiler();
    nnvm::Graph Compile(
        nnvm::Graph graph,
        std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
        std::unordered_map<std::string, int>& arg_dtype_map);

  private:
    UnaryOps create_UnaryOps(const py::module& ns, const py::module& ng);
    BinaryOps create_BinaryOps(const py::module& ns, const py::module& ng);
    void CheckInNGraph(Graph& graph);
    void createPyPlaceholder(NodePtr node, std::string subgraph_name);
    // void createPyOp(NodePtr node);
    void IdentifySubgraphs(Graph& graph);
    void CollapseSubgraphs(Graph& graph);
    void CompileSubgraph(
        std::shared_ptr<Graph> graph,
        std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
        std::unordered_map<std::string, int>& arg_dtype_map);
    void CompileNode(NodePtr node, std::shared_ptr<Graph> graph);
    nnvm::NodeEntry CreateNNVMNode(std::shared_ptr<Graph> graph);
    UnaryOps NgraphUnaryOps_;
    BinaryOps NgraphBinaryOps_;
    std::vector<std::string> NgraphOps_;

    py::module np_;
    py::module ng_;
    py::module ns_;
    py::module ngt_;
    py::object transformer_;

    std::map<std::string, py::object> op_map;
    std::map<std::string, std::vector<std::string> > placeholder_order;

};

} // end namespace ngraph
#endif