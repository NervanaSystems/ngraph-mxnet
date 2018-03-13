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
#include <nnvm/c_api.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/symbolic.h>
#include <algorithm>
#include <cstring>
#include "../executor/exec_pass.h"

#include "../operator/operator_common.h"
#include "ngraph_imperative.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_nnvm_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

// NGImperative constructor for mxnet compute kernel(s)
NGImperative::NGImperative(const nnvm::NodeAttrs &attrs,
                           const mxnet::Context &ctx,
                           const std::vector<mxnet::TBlob> &inputs,
                           const std::vector<mxnet::OpReqType> *req,
                           const std::vector<mxnet::TBlob> &outputs)
    : Compiler(ctx) {
  // Construct nnvm symbol to represent the computation
  auto sym = nnvm::Symbol::CreateFunctor(attrs.op, attrs.dict);
  std::vector<nnvm::Symbol> sym_inputs;
  int icount = 0;
  for (auto i : inputs) {
    sym_inputs.push_back(nnvm::Symbol::CreateVariable(
        attrs.op->name + "_var_" + std::to_string(icount++)));
  }

  // Compose nnvm symbol for compute kernel
  std::vector<const nnvm::Symbol *> psym_inputs;
  std::transform(sym_inputs.begin(), sym_inputs.end(),
                 std::back_inserter(psym_inputs),
                 [](nnvm::Symbol &s) -> const nnvm::Symbol * { return &s; });
  dmlc::array_view<const nnvm::Symbol *> av(psym_inputs);
  std::unordered_map<std::string, const nnvm::Symbol *> tempkwargs;
  sym.Compose(av, tempkwargs, attrs.op->name);

  // construct single symbol nnvm graph and create ngraph
  nnvm::Graph g;
  g.outputs = sym.outputs;
  for (auto i : inputs) {
    shapes_.push_back(i.shape_);
    dtypes_.push_back(i.type_flag_);
  }
  // initialize ngraph
  DeepCopy(g);
  MakeCopiedInputs(sym.ListInputs(nnvm::Symbol::kReadOnlyArgs));
}

// process ngraph composed of nnvm symbol graph
void NGImperative::parse_ngraph() {
  ProcessGraph(NDArrayMap());
  IdentifyCollapseGraphs();
  // imperative assumes graph is just one node
  for (auto n : ngraph_.nodes_)
    if (n->type_ == NodeType::kGraph) {
      // extract and compile subgraph
      op_ngraph_ = compiler_.Compile(n);
      break;
    }
}

// Registers ngraph operators with nnvm
void InitImperativeOnce() {
  static auto &fcompute_cpu =
      nnvm::Op::GetAttr<mxnet::FCompute>("FCompute<cpu>");

  for (auto unique_op : dmlc::Registry<nnvm::Op>::List()) {
    auto op_name = unique_op->name;

    // skip ops not supported by ngraph imperative
    if ((op_name.substr(0, 9) == "_backward") ||
        !NGImperative::check_op_supported(op_name))
      continue;

    nnvm::Op &op =
        ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(op_name);

    // save default mxnet compute kernel for fallback
    auto fallback_fn = fcompute_cpu.get(&op, nullptr);

    // use ngraph immperative, only if fallback available.
    if (fallback_fn) {
      op.set_attr<mxnet::FCompute>(
          "FCompute<cpu>",
          [fallback_fn](const nnvm::NodeAttrs &attrs,
                        const mxnet::OpContext &ctx,
                        const std::vector<mxnet::TBlob> &inputs,
                        const std::vector<mxnet::OpReqType> &req,
                        const std::vector<mxnet::TBlob> &outputs) -> void {
            // thread local cache for ngraph op
            static thread_local NGIOpCache ngicache;

            auto op_key = get_ngiop_key(attrs, ctx.run_ctx.ctx, inputs);
            auto op_ng = ngicache[op_key];
            if (!op_ng) {
              NGImperative ngi(attrs, ctx.run_ctx.ctx, inputs, &req, outputs);
              op_ng = ngicache[op_key] = ngi.get_op_ngraph();
              if (ngraph_log_verbose && op_ng)
                LOG(INFO) << "Caching... " << attrs.op->name;
            }
            int mode = static_cast<int>(GraphExeMode::kInfer);
            if (ctx.is_train) {
              mode = static_cast<int>(GraphExeMode::kTrain);
            }
            if (op_ng && op_ng->ngraph_forward[mode]) {
              compute_forward(ctx, op_ng, inputs, req, outputs);

              if (ngraph_log_verbose_detail) {
                LOG(INFO) << "ngraph imperative op: " << attrs.op->name
                          << ", inputs " << std::to_string(inputs.size())
                          << ", outputs " << std::to_string(outputs.size());

                for (const auto &m : attrs.dict)
                  LOG(INFO) << "attrs.dict[" << m.first << "] = " << m.second;
              }
            } else {
              // use default mxnet compute kernel
              fallback_fn(attrs, ctx, inputs, req, outputs);
            }
          },
          11);
    }
  }
}

void InitImperative() {
  if (!ngraph_gluon_enable) return;
  static std::once_flag onceFlag;
  std::call_once(onceFlag, InitImperativeOnce);
}

size_t NGIOpHash::operator()(const NGIOpKey &key) const {
  std::size_t myhash = std::hash<std::string>()(std::get<0>(key));
  auto ctx = std::get<1>(key);
  myhash = hash_combine(myhash, std::get<0>(ctx));
  myhash = hash_combine(myhash, std::get<1>(ctx));

  auto attrs = std::get<2>(key);
  std::vector<std::string> attrs_keys;
  for (const auto &i : attrs) attrs_keys.push_back(i.first);
  sort(begin(attrs_keys), end(attrs_keys));
  for (const auto &i : attrs_keys) myhash = hash_combine(myhash, i + attrs[i]);

  auto inputs = std::get<3>(key);
  for (const auto &i : inputs) myhash = hash_combine(myhash, i);
  return myhash;
}

bool NGIOpEqual::operator()(const NGIOpKey &t1, const NGIOpKey &t2) const {
  return (std::get<0>(t1) == std::get<0>(t2) &&
          std::get<1>(t1) == std::get<1>(t2) &&
          std::get<2>(t1) == std::get<2>(t2) &&
          std::get<3>(t1) == std::get<3>(t2));
}

NGIOpKey get_ngiop_key(const nnvm::NodeAttrs &attrs, const mxnet::Context &ctx,
                       const std::vector<mxnet::TBlob> &inputs) {
  std::vector<int> in;
  for (const auto &i : inputs) {
    in.push_back(i.type_flag_);
    for (size_t ii = 0; ii < i.shape_.ndim(); ++ii) in.push_back(i.shape_[ii]);
  }
  return NGIOpKey(attrs.op->name, {static_cast<int>(ctx.dev_type),
                                   static_cast<int>(ctx.dev_id)},
                  attrs.dict, in);
}

}  // namespace ngraph_bridge
