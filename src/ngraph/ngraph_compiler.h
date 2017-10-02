#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include <mxnet/ndarray.h>
#include "ngraph_graph.h"
#include "ngraph_sgcompiler.h"

#include <dmlc/any.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/resource.h>
#include <mxnet/op_attr_types.h>

namespace ngraph_bridge {
// map aliases for maps of name, function, where function returns an ngraph
using LayerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;
using NodeMap = std::map<const nnvm::Node*, std::shared_ptr<nnvm::Node>>;
using NNVMNodeVec = std::vector<nnvm::NodePtr>;
using NgraphShape = std::unordered_map<std::string, nnvm::TShape>;
using NgraphDtype = std::unordered_map<std::string, int>;
using NDArrayMap = nnvm::NodeEntryMap<mxnet::NDArray>;
using StateMap = std::unordered_map<const nnvm::Node*, mxnet::OpStatePtr>;

class Compiler {
 public:
  Compiler(const nnvm::Graph& graph,
           const NDArrayMap& feed_dict,
           const NNVMNodeVec& inputs);
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile();
  // parse the nnvm graph into an intermediate rep
  void ParseNNVMGraph();
  StateMap CopySavedStates(const StateMap& saved_states);

  const NgraphShape& GetNgraphShape() { return ngraphShape_; }
  const NgraphDtype& GetNgraphDtype() { return ngraphDtype_; }
  const NDArrayMap& GetFeedDict() { return feedDict_; };
  const NNVMNodeVec& GetInputs() { return inputs_; };

 private:
  // check nodes against ngraph operations
  void CheckInNGraph();
  void DeepCopy(const nnvm::Graph& graph);
  void CopyNodes(const nnvm::Graph& graph);
  void makeCopiedFeedDict(const NDArrayMap& feed_dict);
  void makeCopiedInputs(const NNVMNodeVec& inputs);

  SGCompiler compiler_;
  NodeMap nodeMap_;
  nnvm::Graph graph_;
  ngraph_bridge::Graph ngraph_;
  NgraphShape ngraphShape_;
  NgraphDtype ngraphDtype_;
  nnvm::NodeEntryMap<mxnet::NDArray> feedDict_;
  NNVMNodeVec inputs_;
};

}  // end namespace ngraph
#endif