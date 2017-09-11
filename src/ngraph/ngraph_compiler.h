#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_pycompiler.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

class Compiler {
 public:
  Compiler();
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile(
      nnvm::Graph graph,
      std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
      std::unordered_map<std::string, int>& arg_dtype_map);

 private:
  // Generator to create functions that convert mxnet layer operations
  // into a series of ngraph operations
  layerGraphs create_layerGraphs();
  // parse the nnvm graph into an intermediate rep
  Graph ParseNNVMGraph(nnvm::Graph& graph);
  // check nodes against ngraph operations
  void CheckInNGraph(Graph& graph);
  // create variable objects in ngraph
  // create a new nnvm node based on subgraph
  nnvm::NodeEntry CreateNNVMNode(std::shared_ptr<Graph> graph);
  // vector of available operations
  std::vector<std::string> NgraphOps_;

  PyCompiler compiler_;
};

}  // end namespace ngraph
#endif