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
#include "../../../src/executor/exec_pass.h"

#include "../../../src/operator/operator_common.h"
#include "ngraph_imperative.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_nnvm_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

nnvm::Symbol get_symbol(const nnvm::NodeAttrs &attrs, size_t num_inputs) {
  // Construct nnvm symbol representing this op
  nnvm::Symbol sym;
  auto n = nnvm::Node::Create();
  n->attrs = attrs;
  if (n->op()->attr_parser != nullptr) {
    n->op()->attr_parser(&(n->attrs));
  }
  for (uint32_t i = 0; i < n->num_outputs(); ++i) {
    sym.outputs.emplace_back(nnvm::NodeEntry{n, i, 0});
  }
  std::vector<nnvm::Symbol> sym_inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    sym_inputs.push_back(nnvm::Symbol::CreateVariable(attrs.op->name + "_var_" +
                                                      std::to_string(i)));
  }

  // Compose nnvm symbol for compute kernel
  std::vector<const nnvm::Symbol *> psym_inputs;
  std::transform(sym_inputs.begin(), sym_inputs.end(),
                 std::back_inserter(psym_inputs),
                 [](nnvm::Symbol &s) -> const nnvm::Symbol * { return &s; });
  dmlc::array_view<const nnvm::Symbol *> av(psym_inputs);
  std::unordered_map<std::string, const nnvm::Symbol *> tempkwargs;
  sym.Compose(av, tempkwargs, attrs.op->name);
  return std::move(sym);
}
// NGImperative constructor for mxnet compute kernel(s)
NGImperative::NGImperative(const nnvm::NodeAttrs &attrs,
                           const mxnet::Context &ctx,
                           const std::vector<mxnet::NDArray> &inputs,
                           const std::vector<mxnet::OpReqType> *req,
                           const std::vector<mxnet::NDArray> &outputs)
    : NGImperative(get_symbol(attrs, inputs.size()), ctx, inputs, req,
                   outputs) {}
NGImperative::NGImperative(const nnvm::Symbol &sym, const mxnet::Context &ctx,
                           const std::vector<mxnet::NDArray> &inputs,
                           const std::vector<mxnet::OpReqType> *req,
                           const std::vector<mxnet::NDArray> &outputs)
    : Compiler(ctx) {
  // construct single symbol nnvm graph and create ngraph
  nnvm::Graph g;
  g.outputs = sym.outputs;
  for (auto i : inputs) {
    shapes_.push_back(i.shape());
    dtypes_.push_back(i.dtype());
    stypes_.push_back(mxnet::kDefaultStorage);
  }
  // initialize ngraph
  DeepCopy(g);
  graph_.attrs["context"] = std::make_shared<nnvm::any>(
      mxnet::exec::ContextVector(graph_.indexed_graph().num_nodes(), ctx));
  MakeCopiedInputs(sym.ListInputs(nnvm::Symbol::kReadOnlyArgs));
}
NGImperative::NGImperative(const nnvm::Symbol &sym, const mxnet::Context &ctx,
                           const nnvm::ShapeVector &shapes,
                           const nnvm::DTypeVector &dtypes,
                           const nnvm::StorageVector &stypes)
    : Compiler(ctx) {
  shapes_ = shapes;
  dtypes_ = dtypes;
  stypes_ = stypes;
  // construct single symbol nnvm graph and create ngraph
  nnvm::Graph g;
  g.outputs = sym.outputs;
  // initialize ngraph
  DeepCopy(g);
  graph_.attrs["context"] = std::make_shared<nnvm::any>(
      mxnet::exec::ContextVector(graph_.indexed_graph().num_nodes(), ctx));
  MakeCopiedInputs(sym.ListInputs(nnvm::Symbol::kReadOnlyArgs));
}
NGImperative::NGImperative(const nnvm::Graph &g, const mxnet::Context &ctx)
    : Compiler(ctx) {
  shapes_ = g.GetAttr<nnvm::ShapeVector>("shape");
  dtypes_ = g.GetAttr<nnvm::DTypeVector>("dtype");
  stypes_ = g.GetAttr<mxnet::StorageTypeVector>("storage_type");
  DeepCopy(g);
  nnvm::Symbol sym;
  sym.outputs = g.outputs;
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
// local utility function to cache and invoke ngraph bridge for imperative ops.
// returns true if successfully executed with ngraph, false on failure.
bool compute_forward_imperative(const nnvm::NodeAttrs &attrs,
                                const mxnet::OpContext &ctx,
                                const std::vector<mxnet::NDArray> &inputs,
                                const std::vector<mxnet::OpReqType> &req,
                                const std::vector<mxnet::NDArray> &outputs,
                                std::shared_ptr<Graph> op_ng = nullptr) {
  int mode = ctx.is_train ? static_cast<int>(GraphExeMode::kTrain)
                          : static_cast<int>(GraphExeMode::kInfer);
  if (!sparse_check(inputs) && !sparse_check(outputs)) {
    // thread local cache for ngraph op
    // this allows us to safely operate on cache object
    static thread_local NGIOpCache ngicache;
    auto op_key = get_ngiop_key(attrs, ctx.run_ctx.ctx, inputs);
    op_ng = ngicache[op_key];
    if (!op_ng) {
#ifndef NDEBUG
      if (ngraph_log_verbose_detail) {
        std::cout << "ngraph_imperative: Caching OP " << attrs.op->name
                  << std::endl;
      }
#endif
      NGImperative ngi(attrs, ctx.run_ctx.ctx, inputs, &req, outputs);
      op_ng = ngicache[op_key] = ngi.get_op_ngraph();
    }
  }
  // op_ng can be null if sgcompiler could not create ngraph IR
  // we fallback to mxnet kernel in this case
  if (op_ng && op_ng->ngraph_forward[mode]) {
#ifndef NDEBUG
    // log imperative op details in debug mode
    if (ngraph_log_verbose_detail) {
      std::cout << "ngraph imperative op: " << attrs.op->name << ", inputs "
                << std::to_string(inputs.size()) << ", outputs "
                << std::to_string(outputs.size()) << std::endl;

      for (const auto &m : attrs.dict) {
        std::cout << "attrs.dict[" << m.first << "] = " << m.second
                  << std::endl;
      }
    }
#endif
    compute_forward(ctx, op_ng, inputs, req, outputs);
    return true;
  }
  return false;
}  // namespace ngraph_bridge

struct StateFCompute {
  std::shared_ptr<Graph> ngraph_;
  nnvm::NodeAttrs attrs;
  mxnet::OpStatePtr old_state;
};

// Registers ngraph operators with nnvm
void InitImperativeOnce() {
  static auto &fcomputex_cpu =
      nnvm::Op::GetAttr<mxnet::FComputeEx>("FComputeEx<cpu>");
  static auto &fcompute_cpu =
      nnvm::Op::GetAttr<mxnet::FCompute>("FCompute<cpu>");
  static auto &ndfunc =
      nnvm::Op::GetAttr<mxnet::FNDArrayFunction>("FNDArrayFunction");
  static auto &fscompute_cpu =
      nnvm::Op::GetAttr<mxnet::FStatefulCompute>("FStatefulCompute<cpu>");
  static auto &createop =
      nnvm::Op::GetAttr<mxnet::FCreateOpState>("FCreateOpState");

  for (auto unique_op : dmlc::Registry<nnvm::Op>::List()) {
    auto op_name = unique_op->name;

    // skip ops not supported by ngraph imperative
    if ((op_name.substr(0, 9) == "_backward") ||
        !NGImperative::check_op_supported(op_name)) {
      if (ngraph_log_verbose_detail)
        std::cout << "NGRAPH IMPERATIVE: skipping op -> " << op_name
                  << std::endl;
      continue;
    }

    nnvm::Op &op =
        ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(op_name);

    // save default mxnet compute kernel for fallback
    auto fallbackx_fn = fcomputex_cpu.get(&op, nullptr);
    auto fallback_fn = fcompute_cpu.get(&op, nullptr);
    auto sfallback_fn = fscompute_cpu.get(&op, nullptr);
    auto fallback_nd = ndfunc.get(&op, nullptr);
    auto fallback_st = createop.get(&op, nullptr);

    // use ngraph immperative, only if fallback available.
    if (fallback_nd) {
      op.set_attr<mxnet::FNDArrayFunction>(
          "FNDArrayFunction",
          [fallback_nd](const nnvm::NodeAttrs &attrs,
                        const std::vector<mxnet::NDArray> &inputs,
                        std::vector<mxnet::NDArray> *outputs) -> void {
            const std::vector<mxnet::OpReqType> req(outputs->size());
            if (!compute_forward_imperative(attrs, mxnet::OpContext(), inputs,
                                            req, *outputs)) {
              // use default mxnet compute kernel
              fallback_nd(attrs, inputs, outputs);
            }
          },
          11);
      if (ngraph_log_verbose_detail)
        std::cout << "NGRAPH IMPERATIVE: FNDArrayFunction op -> " << op_name
                  << std::endl;
      continue;
    }

    if (fallbackx_fn) {
      op.set_attr<mxnet::FComputeEx>(
          "FComputeEx<cpu>",
          [fallbackx_fn](const nnvm::NodeAttrs &attrs,
                         const mxnet::OpContext &ctx,
                         const std::vector<mxnet::NDArray> &inputs,
                         const std::vector<mxnet::OpReqType> &req,
                         const std::vector<mxnet::NDArray> &outputs) -> void {
            if (ctx.is_train || ctx.need_grad ||
                !compute_forward_imperative(attrs, ctx, inputs, req, outputs)) {
              // use default mxnet compute kernel
              fallbackx_fn(attrs, ctx, inputs, req, outputs);
            }
          },
          11);
      if (ngraph_log_verbose_detail)
        std::cout << "NGRAPH IMPERATIVE: FComputeEx op -> " << op_name
                  << std::endl;
      continue;
    }
    if (fallback_fn) {
      op.set_attr<mxnet::FCompute>(
          "FCompute<cpu>",
          [fallback_fn](const nnvm::NodeAttrs &attrs,
                        const mxnet::OpContext &ctx,
                        const std::vector<mxnet::TBlob> &inputs,
                        const std::vector<mxnet::OpReqType> &req,
                        const std::vector<mxnet::TBlob> &outputs) -> void {
            std::vector<mxnet::NDArray> in;
            for (auto &i : inputs) in.emplace_back(i, ctx.run_ctx.ctx.dev_id);
            std::vector<mxnet::NDArray> out;
            for (auto &i : outputs) out.emplace_back(i, ctx.run_ctx.ctx.dev_id);
            if (ctx.is_train || ctx.need_grad ||
                !compute_forward_imperative(attrs, ctx, in, req, out)) {
              // use default mxnet compute kernel
              fallback_fn(attrs, ctx, inputs, req, outputs);
            }
          },
          11);
      if (ngraph_log_verbose_detail)
        std::cout << "NGRAPH IMPERATIVE: FCompute op -> " << op_name
                  << std::endl;
      continue;
    }
    // handle legacy FStatefulCompute ops
    if (sfallback_fn) {
      op.set_attr<mxnet::FCreateOpState>(
          "FCreateOpState",
          [fallback_st](const nnvm::NodeAttrs &attrs, mxnet::Context ctx,
                        const std::vector<mxnet::TShape> &in_shape,
                        const std::vector<int> &in_type) -> mxnet::OpStatePtr {
            auto old_state = fallback_st(attrs, ctx, in_shape, in_type);
            auto state_ptr = mxnet::OpStatePtr::Create<StateFCompute>(
                StateFCompute{nullptr, attrs, old_state});
            return state_ptr;
          },
          11);
      op.set_attr<mxnet::FStatefulCompute>(
          "FStatefulCompute<cpu>",
          [sfallback_fn](const mxnet::OpStatePtr &state,
                         const mxnet::OpContext &ctx,
                         const std::vector<mxnet::TBlob> &inputs,
                         const std::vector<mxnet::OpReqType> &req,
                         const std::vector<mxnet::TBlob> &outputs) -> void {
            auto &op_state = state.get_state<StateFCompute>();
            if (!(ctx.is_train || ctx.need_grad)) {
              std::vector<mxnet::NDArray> in;
              for (auto &i : inputs) in.emplace_back(i, ctx.run_ctx.ctx.dev_id);
              std::vector<mxnet::NDArray> out;
              for (auto &i : outputs)
                out.emplace_back(i, ctx.run_ctx.ctx.dev_id);
              if (!op_state.ngraph_) {
                compute_forward_imperative(op_state.attrs, ctx, in, req, out,
                                           op_state.ngraph_);
              } else {
                compute_forward(ctx, op_state.ngraph_, in, req, out);
              }
              // return if ngraph successful
              if (op_state.ngraph_) return;
            }
            // use default mxnet compute kernel
            sfallback_fn(op_state.old_state, ctx, inputs, req, outputs);
          },
          11);
      nnvm::Op &op_stateful_backward =
          ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
              "_backward_" + op_name);
      auto sfallback_bwd_fn = fscompute_cpu.get(&op_stateful_backward, nullptr);
      op_stateful_backward.set_attr<mxnet::FStatefulCompute>(
          "FStatefulCompute<cpu>",
          [sfallback_bwd_fn](const mxnet::OpStatePtr &state,
                             const mxnet::OpContext &ctx,
                             const std::vector<mxnet::TBlob> &inputs,
                             const std::vector<mxnet::OpReqType> &req,
                             const std::vector<mxnet::TBlob> &outputs) -> void {
            auto &op_state = state.get_state<StateFCompute>();
            // use default mxnet compute kernel
            sfallback_bwd_fn(op_state.old_state, ctx, inputs, req, outputs);
          },
          11);
      if (ngraph_log_verbose_detail)
        std::cout << "NGRAPH IMPERATIVE: FStatefulCompute op -> " << op_name
                  << std::endl;
      continue;
    }
    if (ngraph_log_verbose_detail) {
      std::cout << "NGRAPH IMPERATIVE: not implemented -> " << op_name
                << std::endl;
    }
  }
}

void InitImperative() {
  if (!ngraph_gluon_enable()) return;
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
                       const std::vector<mxnet::NDArray> &inputs) {
  std::vector<int> in;
  for (const auto &i : inputs) {
    in.push_back(i.dtype());
    for (size_t ii = 0; ii < i.shape().ndim(); ++ii)
      in.push_back(i.shape()[ii]);
  }
  return NGIOpKey(attrs.op->name, {static_cast<int>(ctx.dev_type),
                                   static_cast<int>(ctx.dev_id)},
                  attrs.dict, in);
}

}  // namespace ngraph_bridge
