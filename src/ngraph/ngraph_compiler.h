#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include <mxnet/ndarray.h>
#include "ngraph_graph.h"
#include "ngraph_pycompiler.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;
using NodeMap = std::map<std::string, std::shared_ptr<nnvm::Node>>;
using nnvmNodeVec = std::vector<nnvm::NodePtr>;
using ngraphShape = std::unordered_map<std::string, nnvm::TShape>;
using ngraphDtype = std::unordered_map<std::string, int>;

class Compiler {
 public:
  Compiler(const nnvm::Graph& graph,
           const nnvm::NodeEntryMap<mxnet::NDArray>& feed_dict,
           std::vector<nnvm::NodePtr> inputs);
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile();
  // parse the nnvm graph into an intermediate rep
  void ParseNNVMGraph();
  nnvm::NodeEntryMap<mxnet::NDArray> GetFeedDict();
  nnvmNodeVec GetInputs();

  const ngraphShape& GetNgraphShape() { return ngraphShape_; }
  const ngraphDtype& GetNgraphDtype() { return ngraphDtype_; }

 private:
  // check nodes against ngraph operations
  void CheckInNGraph();
  void DeepCopy(nnvm::Graph graph);
  void CopyNodes(const nnvm::Graph& graph);
  void makeCopiedFeedDict(nnvm::NodeEntryMap<mxnet::NDArray> feed_dict);
  void makeCopiedInputs(nnvmNodeVec inputs);

  PyCompiler compiler_;
  NodeMap node_map_;
  nnvm::Graph graph_;
  ngraph::Graph ngraph_;
  ngraphShape ngraphShape_;
  ngraphDtype ngraphDtype_;
  nnvm::NodeEntryMap<mxnet::NDArray> feed_dict_;
  nnvmNodeVec inputs_;
};

}  // end namespace ngraph
#endif