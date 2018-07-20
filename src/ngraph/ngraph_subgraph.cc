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

#include <mxnet/ndarray.h>
#include "../operator/subgraph/common.h"
#include "../operator/subgraph/default_subgraph_op.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph_bridge {
using namespace nnvm;
using namespace mxnet;
using namespace mxnet::op;

#define SUBGRAPH_DEBUG 1

std::shared_ptr<ngraph_bridge::Graph>& get_ngraph(const NodeAttrs& attrs) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  auto n = subgraph_sym.outputs[0].node.get();
  return nnvm::get<ngraph_bridge::NGraphParam>(n->attrs.parsed).g;
}

class NgraphSubgraphOperator {
 public:
  explicit NgraphSubgraphOperator(
      std::shared_ptr<ngraph_bridge::Graph> ngraph) {
    ngraph_ = ngraph;
  }

  void Forward(const OpContext& ctx, const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);
  void Backward(const OpContext& ctx, const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs);

 private:
  std::shared_ptr<ngraph_bridge::Graph> ngraph_;
};

void NgraphSubgraphOperator::Forward(const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  compute_forward(ctx, ngraph_, inputs, req, outputs);
}

void NgraphSubgraphOperator::Backward(const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<NDArray>& outputs) {
  compute_backward(ctx, ngraph_, inputs, req, outputs);
}

OpStatePtr CreateNgraphSubgraphOpState(const NodeAttrs& attrs, Context ctx,
                                       const std::vector<TShape>& in_shapes,
                                       const std::vector<int>& in_types) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  auto n = subgraph_sym.outputs[0].node.get();
  return OpStatePtr::Create<NgraphSubgraphOperator>(
      nnvm::get<ngraph_bridge::NGraphParam>(n->attrs.parsed).g);
}

OpStatePtr CreateNgraphBackwardOpState(const NodeAttrs& attrs, Context ctx,
                                       const std::vector<TShape>& in_shapes,
                                       const std::vector<int>& in_types) {
  return OpStatePtr::Create<NgraphSubgraphOperator>(
      nnvm::get<ngraph_bridge::NGraphParam>(attrs.parsed).g);
}

void NgraphSubgraphOpForward(const OpStatePtr& state_ptr, const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  NgraphSubgraphOperator& op = state_ptr.get_state<NgraphSubgraphOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

void NgraphSubgraphOpBackward(const OpStatePtr& state_ptr, const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  NgraphSubgraphOperator& op = state_ptr.get_state<NgraphSubgraphOperator>();
  op.Backward(ctx, inputs, req, outputs);
}

std::vector<nnvm::NodeEntry> NgraphSubgraphGradient(
    const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
  auto graph = get_ngraph(n.get()->attrs);
  const bool zero_grad = check_zero_grad(graph);
  graph->zero_grad = zero_grad;
  bool is_loss = graph->is_loss;
  auto p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward_ngraph_subgraph_op");
  if (!is_loss && zero_grad && p->num_outputs() == 1) {
    return mxnet::op::MakeZeroGradNodes(n, ograds);
  }
  p->attrs.name = n->attrs.name + "_backward";
  p->attrs.dict = n->attrs.dict;
  p->control_deps.emplace_back(n);
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  if (!zero_grad) {
    p->inputs.insert(p->inputs.end(), ograds.begin(), ograds.end());
  }
  p->inputs.insert(p->inputs.end(), n->inputs.begin(), n->inputs.end());
  const Symbol& subgraph_sym = nnvm::get<Symbol>(n.get()->attrs.parsed);
  auto tmpnode = subgraph_sym.outputs[0].node.get();
  p->attrs.parsed =
      std::move(nnvm::get<ngraph_bridge::NGraphParam>(tmpnode->attrs.parsed));
  std::vector<nnvm::NodeEntry> ret;
  for (unsigned i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(nnvm::NodeEntry{p, i, 0});
  }
  return ret;
}

NNVM_REGISTER_OP(_ngraph_subgraph_op)
    .describe(R"code(_ngraph_subgraph_op)code" ADD_FILELINE)
    .set_num_inputs(DefaultSubgraphOpNumInputs)
    .set_num_outputs(DefaultSubgraphOpNumOutputs)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     DefaultSubgraphOpListInputs)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      DefaultSubgraphOpListOutputs)
    .set_attr<FCreateOpState>("FCreateOpState", CreateNgraphSubgraphOpState)
    .set_attr<nnvm::FInferShape>(
        "FInferShape",
        [](const nnvm::NodeAttrs& attrs, std::vector<nnvm::TShape>* in_attrs,
           std::vector<nnvm::TShape>* out_attrs) -> bool {
          auto graph = get_ngraph(attrs);
          std::vector<nnvm::TShape> shapes;
          std::vector<int> dtypes;
          for (auto output : graph->outputs_) {
            shapes.push_back(output->shape_);
            dtypes.push_back(output->dtype_);
          }
          (*out_attrs) = shapes;
          return true;
        })
    .set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
    .set_attr<FInferStorageType>("FInferStorageType",
                                 DefaultSubgraphOpStorageType)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                  NgraphSubgraphOpForward)
    .set_attr<nnvm::FGradient>("FGradient", NgraphSubgraphGradient)
    .set_attr<nnvm::FMutateInputs>(
        "FMutateInputs",
        [](const nnvm::NodeAttrs& attrs) {
          auto graph = get_ngraph(attrs);
          std::vector<uint32_t> mutate_vars;
          for (size_t i = 0; i < graph->inputs_.size(); ++i) {
            if (graph->inputs_[i]->type_ == NodeType::kAux) {
              mutate_vars.emplace_back(i);  // graph->inputs[i]->name);
            }
          }
          return mutate_vars;
        })
    .set_attr<FResourceRequest>("FResourceRequest",
                                DefaultSubgraphOpResourceRequest)
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<FExecType>("FExecType", DefaultSubgraphOpExecType)
    .add_argument("data", "NDArray-or-Symbol[]", "input data list");

NNVM_REGISTER_OP(_backward_ngraph_subgraph_op)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto graph = nnvm::get<ngraph_bridge::NGraphParam>(attrs.parsed).g;
      return graph->num_adjoints_ + graph->inputs_.size();
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto graph = nnvm::get<ngraph_bridge::NGraphParam>(attrs.parsed).g;
      return graph->inputs_.size();
    })
    .set_attr<bool>("TIsBackward", true)
    .set_attr<FExecType>("FExecType", DefaultSubgraphOpExecType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateNgraphBackwardOpState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                  NgraphSubgraphOpBackward)
    .set_attr<FInferStorageType>(
        "FInferStorageType",
        [](const nnvm::NodeAttrs& attrs, const int dev_mask,
           mxnet::DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
           std::vector<int>* out_attrs) {
          return mxnet::op::storage_type_assign(
              out_attrs, mxnet::kDefaultStorage, dispatch_mode,
              mxnet::DispatchMode::kFComputeEx);
        });

}  // namespace ngraph_bridge
