#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include <mxnet/ndarray.h>
#include "ngraph_graph.h"
#include "ngraph_pycompiler.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;
using ngraphShape = std::unordered_map<std::string, nnvm::TShape>;
using ngraphDtype = std::unordered_map<std::string, int>;

class Compiler {
 public:
  Compiler(const nnvm::Graph& graph,
           const nnvm::NodeEntryMap<mxnet::NDArray>& feed_dict);
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile();
  // parse the nnvm graph into an intermediate rep
  void ParseNNVMGraph();

  const ngraphShape& GetNgraphShape() { return ngraphShape_; }
  const ngraphDtype& GetNgraphDtype() { return ngraphDtype_; }

 private:
  // check nodes against ngraph operations
  void CheckInNGraph();

  PyCompiler compiler_;
  nnvm::Graph graph_;
  ngraph::Graph ngraph_;
  ngraphShape ngraphShape_;
  ngraphDtype ngraphDtype_;
};

}  // end namespace ngraph
#endif