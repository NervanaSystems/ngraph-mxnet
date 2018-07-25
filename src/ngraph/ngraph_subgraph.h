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

#include "../operator/subgraph/default_subgraph_op.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph_bridge {
using namespace nnvm;
using namespace mxnet;
using namespace mxnet::op;

class SgNgraphSelector : public SubgraphSelector {
 public:
  SgNgraphSelector() {}

  bool Select(const nnvm::Node &n) override {
    bool match = !n.is_variable() && (n.attrs.name.substr(0, 6) == "ngraph");
    return match;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }
};

class SgNgraphProperty : public SubgraphProperty {
 public:
  SgNgraphProperty() {}

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_ngraph_subgraph_op");
    n->attrs.name = "_ngraph_subgraph_op" + std::to_string(subgraph_id);
    auto tmpnode = sym.outputs[0].node.get();
    n->attrs.parsed =
        nnvm::get<ngraph_bridge::NGraphParam>(tmpnode->attrs.parsed);
    return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgNgraphSelector>();
    return selector;
  }
};

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_
