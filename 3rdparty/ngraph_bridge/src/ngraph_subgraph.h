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

#define DEBUG_SUBGRAPH 0
namespace ngraph_bridge {
using namespace nnvm;
using namespace mxnet;
using namespace mxnet::op;

class SgNgraphSelector : public SubgraphSelector {
 public:
  SgNgraphSelector(Compiler *compiler) : compiler_(compiler) {}

  bool Select(const nnvm::Node &n) override { return is_node_selected(n); }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return is_node_selected(n) && is_node_selected(new_node);
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return is_node_selected(n) && is_node_selected(new_node);
  }
  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) {
    if (candidates.size() == 1 && candidates[0]->inputs.size() == 0) {
      return std::vector<nnvm::Node *>();
    } else {
      return candidates;
    }
  }

 private:
  Compiler *compiler_;
  bool is_node_selected(const nnvm::Node &n) {
    if (compiler_->get_node_map().size() < 1) return false;
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

class SgNgraphProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgNgraphProperty>();
  }

  virtual bool NeedGraphAttrs() const override { return true; }
  virtual nnvm::NodePtr CreateSubgraphNode(
      const nnvm::Symbol &sym, const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_ngraph_subgraph_op");
    n->attrs.name = "_ngraph_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    return n;
  }

  virtual nnvm::NodePtr CreateSubgraphNode(
      const nnvm::Graph &sg, const int subgraph_id = 0) const override {
    nnvm::Symbol sym;
    sym.outputs = sg.outputs;
    auto n = CreateSubgraphNode(sym, subgraph_id);

#if DEBUG_SUBGRAPH
    if (ngraph_log_verbose_detail) {
      nnvm::Graph g;
      g.outputs = sym.outputs;
      auto &orig_graph = GetAttr<nnvm::Graph>("graph");
      std::cout << __func__ << ": id " << subgraph_id
                << " num_nodes after partition "
                << orig_graph.indexed_graph().num_nodes() << "/"
                << g.indexed_graph().num_nodes() << std::endl;
    }
#endif
    auto compiler = std::make_shared<Compiler>(sg);
    compiler->GetNgraph();
    n->attrs.parsed = compiler;
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    if (!compiler_) {
      auto &orig_graph = GetAttr<nnvm::Graph>("graph");
#if DEBUG_SUBGRAPH
      if (ngraph_log_verbose_detail) {
        std::cout << "SgNgraphProperty"
                  << ": Init with orig graph " << &orig_graph << " num_nodes "
                  << orig_graph.indexed_graph().num_nodes() << std::endl;
      }
#endif
      compiler_ = std::make_shared<Compiler>(orig_graph);
    }
#if DEBUG_SUBGRAPH
    if (ngraph_log_verbose_detail) {
      std::cout << __func__ << ": using compiler_ " << compiler_.get()
                << std::endl;
    }
#endif
    return std::make_shared<SgNgraphSelector>(compiler_.get());
  }

 private:
  mutable std::shared_ptr<Compiler> compiler_;
};

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_SUBGRAPH_H_
