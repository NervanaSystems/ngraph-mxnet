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
#ifndef TESTS_CPP_OPERATOR_TEST_SUBGRAPH_API_H_
#define TESTS_CPP_OPERATOR_TEST_SUBGRAPH_API_H_

#include <nnvm/graph.h>

#include <string>
#include <vector>

#include "../../../src/executor/exec_pass.h"
#include "../../../src/operator/nn/concat-inl.h"
#include "../../../src/operator/subgraph/subgraph_property.h"
#include "test_util.h"

class SUBGRAPH_API : public ::testing::Test {
 protected:
  nnvm::NodeEntry createNode(std::string name, std::string op = "") {
    nnvm::NodeAttrs attr;
    auto node = nnvm::Node::Create();
    attr.name = name;
    if (op != "") attr.op = nnvm::Op::Get(op);
    node->attrs = attr;
    nodes_[name] = node;
    return nnvm::NodeEntry{node, 0, 0};
  }

  virtual void SetUp() {
    // set up graph
    auto A = createNode("A");
    auto B = createNode("B");
    auto add1 = createNode("add1", "_add");
    auto add2 = createNode("add2", "_add");
    auto add3 = createNode("add3", "_add");
    auto concat = createNode("concat", "Concat");

    add1.node->inputs.push_back(A);
    add1.node->inputs.push_back(A);

    add2.node->inputs.push_back(add1);
    add2.node->inputs.push_back(A);

    add3.node->inputs.push_back(add2);
    add3.node->inputs.push_back(B);

    for (size_t i = 0; i < 4; ++i) {
      concat.node->inputs.push_back(add3);
    }
    mxnet::op::ConcatParam param;
    param.num_args = concat.node->inputs.size();
    concat.node->attrs.parsed = std::move(std::move(param));

    nnvm_graph.outputs.push_back(concat);

    nnvm::TShape shape{2, 2};

    // Infer storage/dtype/etc
    mxnet::exec::ContextVector contexts(nodes_.size(), mxnet::Context::CPU());
    nnvm::ShapeVector shapes(nodes_.size(), shape);
    nnvm::DTypeVector types(nodes_.size(), 0);
    mxnet::StorageTypeVector stypes(nodes_.size(), mxnet::kDefaultStorage);
    mxnet::exec::DevMaskVector dev_masks(nodes_.size(),
                                         mxnet::Context::CPU().dev_mask());

    nnvm_graph.attrs["context"] =
        std::make_shared<dmlc::any>(std::move(contexts));

    nnvm_graph.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    nnvm_graph = mxnet::exec::InferShape(std::move(nnvm_graph));
    CHECK_EQ(nnvm_graph.GetAttr<size_t>("shape_num_unknown_nodes"), 0U);

    nnvm_graph.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
    nnvm_graph = mxnet::exec::InferType(std::move(nnvm_graph));
    CHECK_EQ(nnvm_graph.GetAttr<size_t>("dtype_num_unknown_nodes"), 0U);

    nnvm_graph.attrs["dev_mask"] =
        std::make_shared<dmlc::any>(std::move(dev_masks));

    nnvm_graph.attrs["storage_type"] =
        std::make_shared<dmlc::any>(std::move(stypes));
    nnvm_graph = mxnet::exec::InferStorageType(std::move(nnvm_graph));

    // set up subgraph_prop
    auto subgraph_prop =
        mxnet::op::SubgraphPropertyRegistry::Get()->CreateSubgraphProperty(
            "default");
    subgraph_prop->SetAttr("op_names",
                           std::unordered_set<std::string>{
                               "_add", "_Plus", "elemwise_add", "_plus",
                           });
    subgraph_prop->SetAttr("graph", nnvm_graph);
    subgraph_prop->SetAttr("grad_reqs", std::vector<mxnet::OpReqType>());
    nnvm_graph.attrs["subgraph_property"] =
        std::make_shared<nnvm::any>(std::move(subgraph_prop));
  }

  virtual void TearDown() {}

  nnvm::Graph nnvm_graph;

  std::unordered_map<std::string, nnvm::NodePtr> nodes_;
};

#endif  // TESTS_CPP_OPERATOR_TEST_SUBGRAPH_API_H_
