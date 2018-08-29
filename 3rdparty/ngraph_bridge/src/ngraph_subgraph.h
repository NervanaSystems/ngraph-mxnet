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

#include "../../../src/operator/subgraph/subgraph_property.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph_bridge {
using namespace nnvm;
using namespace mxnet;
using namespace mxnet::op;

class SgNgraphSelector : public SubgraphSelector {
 public:
  SgNgraphSelector(const nnvm::Graph &g) : compiler_(g) {}

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
  Compiler compiler_;
  bool is_node_selected(const nnvm::Node &n) {
    NodePtr nn;
    MapEntry tmp{&n, 0};
    auto &entry_map = compiler_.get_ngraph().entry_map_;
    if (entry_map.count(tmp)) {
      nn = entry_map[tmp];
    }
    if (nn) {
      return nn->in_ngraph_;
    }
    return false;
  }
};

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
    auto tmpnode = sym.outputs[0].node.get();
    return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    return std::make_shared<SgNgraphSelector>(
        this->GetAttr<nnvm::Graph>("graph"));
  }
};

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_
