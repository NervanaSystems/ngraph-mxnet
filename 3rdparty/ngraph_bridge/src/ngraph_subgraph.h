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

#ifndef MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_
#define MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_

#include "../../../src/operator/subgraph/common.h"
#include "../../../src/operator/subgraph/subgraph_property.h"
#include "ngraph_compiler.h"
#include "ngraph_imperative.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph_bridge {
using namespace nnvm;
using namespace mxnet;
using namespace mxnet::op;

class SgNgraphSelector : public SubgraphSelector {
 public:
  SgNgraphSelector(Compiler *compiler) : compiler_(compiler) {}

  bool Select(const nnvm::Node &n) override {
    return (!n.is_variable() && is_node_selected(n));
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return (!n.is_variable() && is_node_selected(new_node));
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return (!n.is_variable() && is_node_selected(new_node));
  }

 private:
  Compiler *compiler_;
  bool is_node_selected(const nnvm::Node &n) {
    NodePtr nn;
    MapEntry tmp{compiler_->get_node_map().at(&n).get(), 0};
    auto &entry_map = compiler_->get_ngraph().entry_map_;
    if (entry_map.count(tmp)) {
      nn = entry_map[tmp];
    }
    if (nn) {
      return nn->in_ngraph_;
    }
    return false;
  }
};

std::shared_ptr<ngraph_bridge::Graph> create_ngraph(
    const nnvm::NodeAttrs &attrs) {
  const nnvm::Symbol &sym = *attrs.subgraphs[0];
  auto num_inputs = DefaultSubgraphOpNumInputs(attrs);
  auto num_outputs = DefaultSubgraphOpNumOutputs(attrs);
  std::vector<TShape> shapes;
  std::vector<TShape> shapes_out;
  shapes.reserve(num_inputs);
  shapes_out.reserve(num_outputs);
  DefaultSubgraphOpShape(attrs, &shapes, &shapes_out);

  std::vector<int> dtypes;
  std::vector<int> dtypes_out;
  dtypes.reserve(num_inputs);
  dtypes_out.reserve(num_outputs);
  DefaultSubgraphOpType(attrs, &dtypes, &dtypes_out);

  auto ctx = mxnet::Context::CPU();
  std::vector<int> stypes;
  std::vector<int> stypes_out;
  stypes.reserve(num_inputs);
  stypes_out.reserve(num_outputs);
  mxnet::DispatchMode dispatch_mode = DispatchMode::kUndefined;
  DefaultSubgraphOpStorageType(attrs, ctx.dev_mask(), &dispatch_mode, &stypes, &stypes_out);

  NGImperative ngi(sym, ctx, shapes, dtypes, stypes);
  return ngi.get_op_ngraph();
}
class SgNgraphProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgNgraphProperty>();
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_ngraph_subgraph_op");
    n->attrs.name = "_ngraph_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    n->attrs.parsed = create_ngraph(n->attrs);
    return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    if (!compiler_) {
      compiler_ = std::make_shared<Compiler>(GetAttr<nnvm::Graph>("graph"));
    }
    return std::make_shared<SgNgraphSelector>(compiler_.get());
  }

 private:
  mutable std::shared_ptr<Compiler> compiler_;
};

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_
