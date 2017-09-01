#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_utils.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using axes_pair = std::pair<std::string, int>;
using axes_map = std::map<axes_pair, py::object>;
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

using UnaryOps =
    std::map<std::string,
             std::function<py::object(const py::object&, const std::string&)> >;
using BinaryOps =
    std::map<std::string,
             std::function<py::object(const py::object&, const py::object&,
                                      const std::string&)> >;
using LayerOps =
    std::map<std::string,
             std::function<py::object(const NodePtr&, py::object, const std::string&)> >;

class PyCompiler {
 public:
  PyCompiler();
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile(
      nnvm::Graph graph,
      std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
      std::unordered_map<std::string, int>& arg_dtype_map);

 private:
  // create unary operation functions
  UnaryOps create_UnaryOps(const py::module& ng);
  // create binary operation functions
  BinaryOps create_BinaryOps(const py::module& ng);
  // create larger MXNet layer operations
  LayerOps create_LayerOps(const py::module& ng);
  // Generator to create functions that convert mxnet layer operations
  // into a series of ngraph operations
  layerGraphs create_layerGraphs();
  // parse the nnvm graph into an intermediate rep
  Graph ParseNNVMGraph(nnvm::Graph& graph);
  // check nodes against ngraph operations
  void CheckInNGraph(Graph& graph);
  // create variable objects in ngraph
  py::object createPyPlaceholder(std::string name, py::tuple axes);
  // compile subgraph into ngraph python objects
  void CompileSubgraph(std::shared_ptr<Graph> graph);
  // compile inputs to a node
  void CompileInput(NodePtr input, std::string subgraph_name,
                    axes_map node_axes);
  void CompileInputs(NodePtr node, std::string subgraph_name);
  // compile a single node into an ngraph python object
  void CompileNode(NodePtr node, std::shared_ptr<Graph> graph);
  // create a new nnvm node based on subgraph
  nnvm::NodeEntry CreateNNVMNode(std::shared_ptr<Graph> graph);
  py::object make_axis(int length, std::string name);

  // maps of ngraph operation generator functions
  UnaryOps NgraphUnaryOps_;
  BinaryOps NgraphBinaryOps_;
  LayerOps NgraphLayerOps_;

  // vector of available operations
  std::vector<std::string> NgraphOps_;

  // python modules and objects required for compilation/computation
  py::module ng_;
  py::module ngt_;
  py::object transformer_;

  // information on how
  std::map<std::string, py::object> op_map;
  std::map<std::string, std::vector<std::string> > placeholder_order;
};

}  // end namespace ngraph
#endif