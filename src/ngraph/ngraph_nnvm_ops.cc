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
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/symbolic.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../operator/operator_common.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_nnvm_utils.h"

namespace ngraph_bridge {

// get the OP from nnvm, return a pointer to it.
nnvm::Op *get_subgraph_op(std::shared_ptr<Graph> graph) {
  return &(::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
      "ngraph_" + graph->name_));
}

void append_cached_to_forward(TensorViewVector *results,
                              const std::shared_ptr<Graph> &graph,
                              const int mode) {
  if (results == nullptr) {
    throw std::runtime_error(
        "NGRAPH_BRIDGE: append_cached_to_forward recieved nullptr results");
  }
  results->insert(results->end(), graph->cached_aux_values[mode].begin(),
                  graph->cached_aux_values[mode].end());
  results->insert(results->end(), graph->cached_values[mode].begin(),
                  graph->cached_values[mode].end());
}

// function for computing forward on ngraph
void compute_forward(const mxnet::OpContext &ctx, std::shared_ptr<Graph> graph,
                     const std::vector<mxnet::NDArray> &inputs,
                     const std::vector<mxnet::OpReqType> &req,
                     const std::vector<mxnet::NDArray> &outputs) {
  auto backend = GetBackendFromContext(graph->context_);
  auto placeholders = make_ngraph_placeholders(inputs, backend, true);
  auto results = make_ngraph_placeholders(outputs, backend, false);

  int mode = static_cast<int>(GraphExeMode::kInfer);
  if (ctx.is_train) {
    mode = static_cast<int>(GraphExeMode::kTrain);
    graph->forward_train_computed = true;
  }
  assert(graph->ngraph_forward[mode] != nullptr);
  append_cached_to_forward(&results, graph, mode);
  graph->ngraph_forward[mode]->call(results, placeholders);

  std::vector<mxnet::NDArray> outs = {outputs[0]};
  result_to_NDArray(results, req, outs);
}

// function for computing backward on ngraph
void compute_backward(const mxnet::OpContext &ctx, std::shared_ptr<Graph> graph,
                      const std::vector<mxnet::NDArray> &inputs,
                      const std::vector<mxnet::OpReqType> &req,
                      const std::vector<mxnet::NDArray> &outputs) {
  // only expect backward is called in training mode
  assert(ctx.is_train);
  auto backend = GetBackendFromContext(graph->context_);

  const int mode = static_cast<int>(GraphExeMode::kTrain);

  // check forward has been executed, if not we need to run forward to
  // generate valid data in fprop cache
  if (graph->enable_fprop_cache && !graph->forward_train_computed) {
    // forward inputs
    std::vector<mxnet::NDArray> fwd_inputs(inputs.begin() + graph->num_outputs,
                                           inputs.end());
    auto placeholders = make_ngraph_placeholders(fwd_inputs, backend, true);
    // forward outputs
    auto shape = TShape_to_NShape(graph->nodes_.back()->shape_);
    const auto &element_type = getType(graph->nodes_.back()->dtype_);
    auto output_tv = backend->make_primary_tensor_view(element_type, shape);
    TensorViewVector results{output_tv};
    append_cached_to_forward(&results, graph, mode);
    // call forward
    graph->ngraph_forward[mode]->call(results, placeholders);
  }

  // backward op
  auto placeholders = graph->enable_fprop_cache
                          ? make_ngraph_placeholders({inputs[0]}, backend, true)
                          : make_ngraph_placeholders(inputs, backend, true);

  auto results = make_ngraph_placeholders(outputs, backend, false);
  placeholders.insert(placeholders.end(), graph->cached_values[mode].begin(),
                      graph->cached_values[mode].end());

  CHECK(graph->ngraph_backward[mode]);
  graph->ngraph_backward[mode]->call(results, placeholders);
  // reset the forward training compute flag to ensure backward always have
  // updated data from forward
  graph->forward_train_computed = false;
  result_to_NDArray(results, req, outputs);

  // overwrite aux data if they exist
  // aux result outputs mapped to inputs
  const size_t cached_aux_count = graph->cached_aux_values[mode].size();
  if (cached_aux_count > 0) {
    std::vector<mxnet::OpReqType> aux_req;
    std::vector<mxnet::NDArray> aux_outs;

    for (size_t i = 0; i < cached_aux_count; ++i) {
      aux_outs.push_back(inputs[graph->cached_aux_positions[mode][i] + 1]);
      aux_req.push_back(mxnet::kWriteTo);
    }

    result_to_NDArray(graph->cached_aux_values[mode], aux_req, aux_outs);
  }
}

// check if last node in graph is an op that doesnt need head-gradient
bool check_zero_grad(const std::shared_ptr<Graph> &graph) {
  auto size = graph->nodes_.size();
  if ((size < 1) || (graph->nodes_[size - 1]->type_ != NodeType::kOp))
    return false;
  if (ops_no_head_grad.count(graph->nodes_[size - 1]->operation_)) return true;
  return false;
}

void register_forward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto &op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
      "ngraph_" + graph->name_);
  // setup the inputs and outpus
  int num_inputs = graph->inputs_.size();
  op.set_num_inputs(num_inputs);
  op.set_num_outputs(1);

  // register the inputs with nnvm
  std::vector<std::string> input_names;
  for (auto n : graph->inputs_) {
    input_names.emplace_back(n->name_);
    op.add_argument(n->name_, "NDArray-or-Symbol", "argument " + n->name_);
  }

  // register lambda to list inputs
  op.set_attr<nnvm::FListInputNames>(
      "FListInputNames",
      [input_names](const nnvm::NodeAttrs &attrs) { return input_names; });

  // // get the auxillary inputs
  std::vector<uint32_t> mutate_vars;
  for (size_t i = 0; i < graph->inputs_.size(); ++i) {
    if (graph->inputs_[i]->type_ == NodeType::kAux) {
      mutate_vars.emplace_back(i);  // graph->inputs[i]->name);
    }
  }

  // register lambda to list inputs
  op.set_attr<nnvm::FMutateInputs>(
      "FMutateInputs",
      [mutate_vars](const nnvm::NodeAttrs &attrs) { return mutate_vars; });

  // dummy attribute parser for execution
  auto attr_parser = [](nnvm::NodeAttrs *attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      attrs->parsed = std::move(op);
    }
  };
  op.set_attr_parser(attr_parser);

  // register lambda to say nothing is inplace
  op.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [num_inputs](const nnvm::NodeAttrs &attrs) {
                                      std::vector<std::pair<int, int>> inplace;
                                      for (int i = 0; i < num_inputs; ++i)
                                        inplace.push_back({i, 0});
                                      return inplace;
                                    });

  // register another lambda to say nothing is in place
  op.set_attr<nnvm::FInplaceIdentity>(
      "FInplaceIdentity", [num_inputs](const nnvm::NodeAttrs &attrs) {
        std::vector<bool> inplace;
        for (int i = 0; i < num_inputs; ++i) inplace.push_back(false);
        return inplace;
      });

  // Register Gradient node generation function
  // check if zero grad
  const bool zero_grad = check_zero_grad(graph);
  auto back_op_name = "_backward_" + ("ngraph_" + graph->name_);
  op.set_attr<nnvm::FGradient>(
      "FGradient",
      [back_op_name, zero_grad](const nnvm::NodePtr &n,
                                const std::vector<nnvm::NodeEntry> &ograds) {
        auto p = nnvm::Node::Create();
        p->attrs.op = nnvm::Op::Get(back_op_name);
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
        std::vector<nnvm::NodeEntry> ret;
        for (unsigned i = 0; i < p->num_outputs(); ++i) {
          ret.emplace_back(nnvm::NodeEntry{p, i, 0});
        }
        return ret;
      });

  // This is bad. need to redo
  // currently just returing the data types and shapes of the output nodes
  // this subgraph is replacing that were inferred by mxnet
  // not currently checking with the ngraph operations to see if they
  // return the same shape
  auto shape = graph->shape_;
  auto dtype = graph->dtype_;
  op.set_attr<nnvm::FInferShape>(
      "FInferShape",
      [shape](const nnvm::NodeAttrs &attrs, std::vector<nnvm::TShape> *in_attrs,
              std::vector<nnvm::TShape> *out_attrs) -> bool {
        (*out_attrs)[0] = shape;
        return true;
      });

  // similarly bad
  op.set_attr<nnvm::FInferType>(
      "FInferType",
      [dtype](const nnvm::NodeAttrs &attrs, std::vector<int> *iattr,
              std::vector<int> *oattr) -> bool {
        return mxnet::op::type_assign(&((*oattr)[0]), dtype);
      });

  op.set_attr<mxnet::FInferStorageType>(
      "FInferStorageType",
      [](const nnvm::NodeAttrs &attrs, const int dev_mask,
         mxnet::DispatchMode *dispatch_mode, std::vector<int> *in_attrs,
         std::vector<int> *out_attrs) {
        return mxnet::op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                              dispatch_mode,
                                              mxnet::DispatchMode::kFComputeEx);
      });
  // create the compute lambda
  op.set_attr<mxnet::FComputeEx>(
      "FComputeEx<cpu>",
      [graph](const nnvm::NodeAttrs &attrs, const mxnet::OpContext &ctx,
              const std::vector<mxnet::NDArray> &inputs,
              const std::vector<mxnet::OpReqType> &req,
              const std::vector<mxnet::NDArray> &outputs) -> void {
        compute_forward(ctx, graph, inputs, req, outputs);
      });
}

void register_backward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto &op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
      "_backward_" + ("ngraph_" + graph->name_));
  // setup the inputs and outpus
  int num_inputs = graph->inputs_.size();
  op.set_num_inputs(num_inputs + 1);
  op.set_num_outputs(num_inputs);

  // Mark as backward
  op.set_attr<bool>("TIsBackward", true);

  op.set_attr<mxnet::FInferStorageType>(
      "FInferStorageType",
      [](const nnvm::NodeAttrs &attrs, const int dev_mask,
         mxnet::DispatchMode *dispatch_mode, std::vector<int> *in_attrs,
         std::vector<int> *out_attrs) {
        return mxnet::op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                              dispatch_mode,
                                              mxnet::DispatchMode::kFComputeEx);
      });
  // create the compute lambda
  op.set_attr<mxnet::FComputeEx>(
      "FComputeEx<cpu>",
      [graph](const nnvm::NodeAttrs &attrs, const mxnet::OpContext &ctx,
              const std::vector<mxnet::NDArray> &inputs,
              const std::vector<mxnet::OpReqType> &req,
              const std::vector<mxnet::NDArray> &outputs) -> void {
        compute_backward(ctx, graph, inputs, req, outputs);
      });
}
// register subgraph ops with nnvm.
void register_subgraph(std::shared_ptr<Graph> graph) {
  register_forward_op(graph);
  register_backward_op(graph);
}

}  // namespace ngraph_bridge
