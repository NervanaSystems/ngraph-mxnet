#ifndef NGRAPH_PYCOMPILER_H_
#define NGRAPH_PYCOMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_utils.h"
#include "ngraph_pyemitter.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using axes_pair = std::pair<std::string, int>;
using axes_map = std::map<axes_pair, py::object>;
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

class PyCompiler : public PyEmitter {
 public:
  PyCompiler(){};
  std::shared_ptr<Graph> Compile(NodePtr graph);
 private:
  // Generator to create functions that convert mxnet layer operations
  // into a series of ngraph operations
  layerGraphs create_layerGraphs();
  // parse the nnvm graph into an intermediate rep
  Graph ParseNNVMGraph(nnvm::Graph& graph);
  // check nodes against ngraph operations
  void CheckInNGraph(Graph& graph);
  // compile subgraph into ngraph python objects
  void CompileSubgraph(std::shared_ptr<Graph> graph);
  // compile inputs to a node
  void CompileInput(NodePtr input, std::string subgraph_name,
                    axes_map node_axes);
  void CompileInputs(NodePtr node, std::string subgraph_name);
  // compile a single node into an ngraph python object
  void CompileNode(NodePtr node, std::shared_ptr<Graph> graph);
};

}  // end namespace ngraph
#endif