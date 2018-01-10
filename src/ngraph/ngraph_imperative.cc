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
  CollapseSubgraphs(&ngraph_);
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
    if (!NGImperative::check_op_supported(op_name)) continue;

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
            NGImperative ngi(attrs, ctx.run_ctx.ctx, inputs, &req, outputs);
            auto op_ng = ngi.get_op_ngraph();
            if (op_ng && op_ng->ngraph_forward) {
              compute_forward(op_ng, inputs, outputs);

// TODO(aemani): refactor using mxnet verbose log
// convenient debug utility.
#if 0
                std::cout << "ngraph imperative op: " << attrs.op->name
                          << ", inputs " << std::to_string(inputs.size())
                          << ", outputs " << std::to_string(outputs.size())
                          << std::endl;
                for (const auto &m : attrs.dict)
                  std::cout << "attrs.dict[" << m.first << "] = " << m.second
                            << '\n';
#endif
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
  static std::once_flag onceFlag;
  std::call_once(onceFlag, InitImperativeOnce);
}

}  // namespace ngraph_bridge
