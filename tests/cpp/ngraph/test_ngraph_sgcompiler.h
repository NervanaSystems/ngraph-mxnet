// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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
#ifndef TESTS_CPP_NGRAPH_TEST_NGRAPH_SGCOMPILER_H_
#define TESTS_CPP_NGRAPH_TEST_NGRAPH_SGCOMPILER_H_

#include <vector>

#include "test_util.h"

#include "../../src/ngraph/ngraph_sgcompiler.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

class NGRAPH_SGCOMPILER : public ::testing::Test {
 protected:
  NodePtr in1;
  NodePtr in2;
  NodePtr in3;
  NodePtr node1;
  NodePtr node2;
  std::shared_ptr<Graph> subgraph;
  virtual void SetUp() {
    const auto shape = nnvm::TShape{4, 8, 12, 16};

    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");
    in3 = std::make_shared<VariableNode>(nullptr, "in3");

    node1 = std::make_shared<OpNode>(nullptr, "node1", "_plus",
                                     std::vector<NodePtr>{in1, in2});
    node2 = std::make_shared<OpNode>(nullptr, "node2", "_plus",
                                     std::vector<NodePtr>{node1, in3});

    in1->shape_ = shape;
    in2->shape_ = shape;
    in3->shape_ = shape;
    node1->shape_ = shape;
    node2->shape_ = shape;

    subgraph = std::make_shared<Graph>();
    subgraph->inputs_.push_back(in1);
    subgraph->inputs_.push_back(in2);
    subgraph->inputs_.push_back(in3);
    subgraph->nodes_.push_back(node1);
    subgraph->nodes_.push_back(node2);
  }

  virtual void TearDown() {}
};

class testSGCompiler : public SGCompiler {
 public:
  using SGCompiler::Compile;
  using SGCompiler::CompileNodes;
  using SGCompiler::ngraph_op_funcs_;
  using SGCompiler::op_map_;

  std::shared_ptr<ngraph::Node> operator[](NodePtr node) {
    return op_map_[node];
  }

  int count(NodePtr node) { return op_map_.count(node); }
};

}  // namespace ngraph_bridge

#endif  // TESTS_CPP_NGRAPH_TEST_NGRAPH_SGCOMPILER_H_
