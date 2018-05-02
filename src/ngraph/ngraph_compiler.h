/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MXNET_NGRAPH_NGRAPH_COMPILER_H_
#define MXNET_NGRAPH_NGRAPH_COMPILER_H_

#include <dmlc/any.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/resource.h>
#include <mxnet/tensor_blob.h>

#include <map>
#include <string>
#include <vector>

#include "ngraph_graph.h"
#include "ngraph_sgcompiler.h"
#include "nnvm/graph_attr_types.h"

namespace ngraph_bridge {

// Aliases to simplify code
using LayerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;
using NodeMap = std::map<const nnvm::Node*, std::shared_ptr<nnvm::Node>>;
using NNVMNodeVec = std::vector<nnvm::NodePtr>;
using NgraphShape = std::unordered_map<std::string, nnvm::TShape>;
using NgraphDType = std::unordered_map<std::string, int>;
using NgraphSType = std::unordered_map<std::string, int>;
using NDArrayMap = nnvm::NodeEntryMap<mxnet::NDArray>;
using StateMap = std::unordered_map<const nnvm::Node*, mxnet::OpStatePtr>;

// This struct collects arguments and provides a method for
// ngraph_bridge::Compiler to infer nnvm::Graph shape and dtype
// prior to compilation of the ngraph.  There are two flavos to
// consider, Bind and SimpleBind, matching the two flavors of
// GraphExecutor::Init function where ngraph_bridge::Compiler is
// invoked.  Hence there are two derivations of this base object.
struct BindArgBase {
  explicit BindArgBase(size_t numforward) : kNumForwardInputs(numforward) {}
  virtual ~BindArgBase() {}

  // common arguments
  const size_t kNumForwardInputs;
};

// Bind
struct BindArg : public BindArgBase {
  BindArg(size_t numforward, const std::vector<mxnet::NDArray>& inargs,
          const std::vector<mxnet::NDArray>& auxstates)
      : BindArgBase(numforward), in_args_(inargs), aux_states_(auxstates) {}

  // bind arguments
  const std::vector<mxnet::NDArray> in_args_;
  const std::vector<mxnet::NDArray> aux_states_;
};

// SimpleBind
struct SimpleBindArg : public BindArgBase {
  SimpleBindArg(size_t numforward,
                const std::unordered_map<std::string, nnvm::TShape>& shapes,
                const std::unordered_map<std::string, int>& dtypes,
                const std::unordered_map<std::string, int>& stypes)
      : BindArgBase(numforward),
        shape_map_(shapes),
        dtype_map_(dtypes),
        stype_map_(stypes) {}

  // simple bind arguments
  const NgraphShape shape_map_;
  const NgraphDType dtype_map_;
  const NgraphDType stype_map_;
};

// This is a compile-time hash map that contains information on
// nnvm alias renaming to simplify the emitter class -
// we don't want to emit _Plus, _plus, _add, and elemwise_add
// for the same op
static std::unordered_map<std::string, std::string> nameswitch({
    // elemwise
    {"elemwise_add", "_plus"},
    {"elemwise_sub", "_minus"},
    {"elemwise_mul", "_mul"},
    {"elemwise_div", "_div"},
    // broadcast
    {"broadcast_plus", "broadcast_add"},
    {"broadcast_minus", "broadcast_sub"},
    // scalar
    {"_PlusScalar", "_plus_scalar"},
    {"_MinusScalar", "_minus_scalar"},
    {"_RMinusScalar", "_rminus_scalar"},
    {"_MulScalar", "_mul_scalar"},
    {"_DivScalar", "_div_scalar"},
    {"_RDivScalar", "_rdiv_scalar"},
    {"_EqualScalar", "_equal_scalar"},
    {"_NotEqualScalar", "_not_equal_scalar"},
    {"_RMinusScalar", "_rminus_scalar"},
    {"_GreaterScalar", "_greater_scalar"},
    {"_GreaterEqualScalar", "_greater_equal_scalar"},
    {"_LesserScalar", "_lesser_scalar"},
    {"_LesserEqualScalar", "_lesser_equal_scalar"},
    // Binary Basic
    {"_add", "_plus"},
    {"_Plus", "_plus"},
    {"_sub", "_minus"},
    {"_Minus", "_minus"},
    {"_Mul", "_mul"},
    {"_Div", "_div"},
    {"_Mod", "_mod"},
    // Binary Extended
    {"_Power", "_power"},
    {"_Maximum", "_maximum"},
    {"_Minimum", "_minimum"},
    {"_Hypot", "_hypot"},
    // Binary Logic
    {"_Equal", "_equal"},
    {"_Not_Equal", "_not_equal"},
    {"_Greater", "_greater"},
    {"_Greater_Equal", "_greater_equal"},
    {"_Lesser", "_lesser"},
    {"_Lesser_Equal", "_lesser_equal"},
    // Layer Ops
    {"Concat", "concat"},
    {"Flatten", "flatten"},
    // Unary Ops
    {"Reshape", "reshape"},
    {"SwapAxis", "swapaxes"},
    {"Cast", "cast"},
    {"sum_axis", "sum"},
    {"SliceChannel", "split"},
});

// MxNet OPs that do not have gradient should work when head-gradient is not
// passed. Below OPs in ngrap-bridge fit this criteria.
static std::unordered_set<std::string> ops_no_head_grad{
    "_equal",
    "_not_equal",
    "_greater",
    "_greater_equal",
    "_lesser",
    "_lesser_equal",
    "broadcast_equal",
    "broadcast_not_equal",
    "broadcast_greater",
    "broadcast_greater_equal",
    "broadcast_lesser",
    "broadcast_lesser_equal",
    "_equal_scalar",
    "_not_equal_scalar",
    "_greater_scalar",
    "_greater_equal_scalar",
    "_lesser_scalar",
    "_lesser_equal_scalar"};

// Utility function for replacing operation names
// based on the dict above
inline std::string clean_opname(std::string name) {
  if (nameswitch.count(name)) {
    return nameswitch[name];
  } else {
    return name;
  }
}
// Main compiler class
class Compiler {
 public:
  // Construction takes setup from the grad executor and preps the graph
  // for ngraph compilation
  Compiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
           const NNVMNodeVec& inputs, const BindArgBase& bindarg,
           const mxnet::Context& context);
  // Construct base compiler object with context only
  Compiler(const mxnet::Context& context);
  // Constructor for use with gluon hybridize
  Compiler(const nnvm::Graph& graph, const mxnet::Context& context,
           const std::vector<nnvm::TShape> shapes,
           const std::vector<int> dtypes, const std::vector<int> stypes);
  // Compile returns the compiled graph
  nnvm::Graph Compile();
  // parse the nnvm graph into an intermediate represenation
  // TODO(mbrookhart): Make this protected, it's here for debugging
  void ParseNnvmGraph();

  StateMap CopySavedStates(const StateMap& saved_states);
  // Return maps of the shapes and dtypes for further analysis in graph_executor
  const NgraphShape& GetNgraphShape() { return ngraph_shape_; }
  const NgraphDType& GetNgraphDtype() { return ngraph_dtype_; }
  const NgraphSType& GetNgraphStype() { return ngraph_stype_; }
  // Return copies of the feed_dict and inputs to feed back into the
  // graph executor inference engine
  const NDArrayMap& GetFeedDict() { return feed_dict_; }
  const NNVMNodeVec& GetInputs() { return inputs_; }

 protected:
  // parse and process graph
  void ProcessGraph(const NDArrayMap& feed_dict);
  // check nodes against ngraph operations
  void CheckInNgraph();
  // make a deep copy of the graph and graph nodes
  void DeepCopy(const nnvm::Graph& graph);
  // copy nodes outside of the graph
  void CopyNodes(const nnvm::Graph& graph);
  // create a copied representaiton of the feed_dict
  void MakeCopiedFeedDict(const NDArrayMap& feed_dict);
  // create a copied representation of the inputs
  void MakeCopiedInputs(const NNVMNodeVec& inputs);

  void IdentifyCollapseGraphs();

  void CreateSubgraphNNVMNodes();
  void ConnectSubgraphNodes();
  void CollapseNNVMGraph();
  void CleanUpUneededReferences();
  // class to compile subgraphs
  SGCompiler compiler_;
  // storage for copied nodes
  NodeMap node_map_;
  // storage for copied graph
  nnvm::Graph graph_;
  // storage for IR graph
  ngraph_bridge::Graph ngraph_;
  // shape, type and storage_type maps to return to the graph executor
  NgraphShape ngraph_shape_;
  NgraphDType ngraph_dtype_;
  NgraphDType ngraph_stype_;
  // copied feed dict and inputs
  nnvm::NodeEntryMap<mxnet::NDArray> feed_dict_;
  NNVMNodeVec inputs_;

  // infer nnvm::Graph shape and type for bind case
  void Infer(const BindArg* bind);
  // infer nnvm::Graph shape and type for simple bind case
  void Infer(const SimpleBindArg* simplebind);
  // infer nnvm::Graph shape and type for simple bind case
  void InferGraphProperties();

  // inferred nnvm::Graph shape
  nnvm::ShapeVector shapes_;
  // inferred nnvm::Graph dtype
  nnvm::DTypeVector dtypes_;
  // inferred nnvm::Graph storage type
  nnvm::StorageVector stypes_;
  std::unordered_map<std::shared_ptr<Graph>, nnvm::NodePtr> compiled_nodes_;
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_COMPILER_H_
