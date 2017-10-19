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

#include "test_util.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler.h"

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
    const auto shape = nnvm::TShape{4,8,12,16};

    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");
    in3 = std::make_shared<VariableNode>(nullptr, "in3");

    node1 = std::make_shared<OpNode>(nullptr, "node1", "_plus",
                                     std::vector<NodePtr>{in1, in2});
    node2 = std::make_shared<OpNode>(nullptr, "node2", "_plus",
                                     std::vector<NodePtr>{node1, in3});

    in1->shape = shape;
    in2->shape = shape;
    in3->shape = shape;
    node1->shape = shape;
    node2->shape = shape;

    subgraph = std::make_shared<Graph>();
    subgraph->inputs.push_back(in1);
    subgraph->inputs.push_back(in2);
    subgraph->inputs.push_back(in3);
    subgraph->nodes_.push_back(node1);
    subgraph->nodes_.push_back(node2);
  }

  virtual void TearDown(){};
};

class testSGCompiler : public SGCompiler {
  public:
    using SGCompiler::op_map;
    using SGCompiler::NgraphOpFuncs_;
    using SGCompiler::CompileInput;
    using SGCompiler::CompileNode;
    using SGCompiler::Compile;

    std::shared_ptr<ngraph::Node> operator[](NodePtr node){
      return op_map[node];
    }

    int count(NodePtr node){
      return op_map.count(node);
    }

};

}