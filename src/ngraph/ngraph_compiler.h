// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include <mxnet/ndarray.h>
#include "ngraph_graph.h"
#include "ngraph_sgcompiler.h"

#include <dmlc/any.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/resource.h>
#include <mxnet/tensor_blob.h>
#include "nnvm/graph_attr_types.h"

namespace ngraph_bridge {
// map aliases for maps of name, function, where function returns an ngraph
using LayerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;
using NodeMap = std::map<const nnvm::Node*, std::shared_ptr<nnvm::Node>>;
using NNVMNodeVec = std::vector<nnvm::NodePtr>;
using NgraphShape = std::unordered_map<std::string, nnvm::TShape>;
using NgraphDType = std::unordered_map<std::string, int>;
using NDArrayMap = nnvm::NodeEntryMap<mxnet::NDArray>;
using StateMap = std::unordered_map<const nnvm::Node*, mxnet::OpStatePtr>;

// This struct collects arguments and provides a method for
// ngraph_bridge::Compiler to infer nnvm::Graph shape and dtype
// prior to compilation of the ngraph.  There are two flavos to
// consider, Bind and SimpleBind, matching the two flavors of
// GraphExecutor::Init function where ngraph_bridge::Compiler is
// invoked.  Hence there are two derivations of this base object.
struct BindArgBase {
  BindArgBase(size_t numforward) : numForwardInputs_(numforward) {}
  virtual ~BindArgBase() {}

  // common arguments
  const size_t numForwardInputs_;
};

// Bind
struct BindArg : public BindArgBase {
  BindArg(size_t numforward, const std::vector<mxnet::NDArray>& inargs,
          const std::vector<mxnet::NDArray>& auxstates)
      : BindArgBase(numforward), inArgs_(inargs), auxStates_(auxstates) {}

  // bind arguments
  const std::vector<mxnet::NDArray> inArgs_;
  const std::vector<mxnet::NDArray> auxStates_;
};

// SimpleBind
struct SimpleBindArg : public BindArgBase {
  SimpleBindArg(size_t numforward,
                const std::unordered_map<std::string, nnvm::TShape>& shapes,
                const std::unordered_map<std::string, int>& dtypes)
      : BindArgBase(numforward), shapeMap_(shapes), dtypeMap_(dtypes) {}

  // simple bind arguments
  const NgraphShape shapeMap_;
  const NgraphDType dtypeMap_;
};

static std::unordered_map<std::string, std::string> nameswitch({
  // elemwise
  {"elemwise_add", "_plus"},
  {"elemwise_sub", "_minus"},
  {"elemwise_mul", "_mul"},
  {"elemwise_div", "_div"},
  // broadcast
  {"broadcast_plus", "broadcast_add"},
  {"broadcast_minus", "broadcast_sub"},
  //Binary Basic
  {"_add", "_plus"},
  {"_Plus", "_plus"},
  {"_sub", "_minus"},
  {"_Minus", "_minus"},
  {"_Mul", "_mul"},
  {"_Div", "_div"},
  {"_Mod", "_mod"},
  //Binary Extended
  {"_Power","_power"},
  {"_Maximum","_maximum"},
  {"_Minimum","_minimum"},
  {"_Hypot","_hypot"},
  //Binary Logic
  {"_Equal", "_equal"},
  {"_Not_Equal", "_not_equal"},
  {"_Greater", "_greater"},
  {"_Greater_Equal", "_greater_equal"},
  {"_Lesser", "_lesser"},
  {"_Lesser_Equal", "_lesser_equal"},
  //Layer Ops
  {"Concat", "concat"},
});

inline std::string clean_opname(std::string name) {
  if (nameswitch.count(name)){
    return nameswitch[name];
  } else {
    return name;
  }
}

class Compiler {
 public:
  Compiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
           const NNVMNodeVec& inputs, const BindArgBase& bindarg);
  // Main interface from MXNet
  // Compile a graph, take an MXNet graph and replace subsections of it
  // with ngraph operations
  nnvm::Graph Compile();
  // parse the nnvm graph into an intermediate rep
  void ParseNNVMGraph();
  StateMap CopySavedStates(const StateMap& saved_states);

  const NgraphShape& GetNgraphShape() { return ngraphShape_; }
  const NgraphDType& GetNgraphDtype() { return ngraphDtype_; }
  const NDArrayMap& GetFeedDict() { return feedDict_; };
  const NNVMNodeVec& GetInputs() { return inputs_; };

 protected:
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
  NgraphDType ngraphDtype_;
  nnvm::NodeEntryMap<mxnet::NDArray> feedDict_;
  NNVMNodeVec inputs_;

  // infer nnvm::Graph shape and type for bind case
  void Infer(const BindArg* bind);
  // infer nnvm::Graph shape and type for simple bind case
  void Infer(const SimpleBindArg* simplebind);

  // inferred nnvm::Graph shape
  nnvm::ShapeVector shapes_;
  // inferred nnvm::Graph dtype
  nnvm::DTypeVector dtypes_;
};

}  // end namespace ngraph
#endif