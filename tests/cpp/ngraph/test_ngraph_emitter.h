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

#include "../../src/ngraph/ngraph_emitter.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

struct testGeneralEmitter : public Emitter, public ::testing::Test {
protected:
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<VariableNode> in3;
  std::shared_ptr<OpNode> node;

  NgraphNodePtr data1;
  NgraphNodePtr data2;
  NgraphNodePtr data3;

  virtual void SetUp() {
    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");
    in3 = std::make_shared<VariableNode>(nullptr, "in3");

    node = std::make_shared<OpNode>(nullptr, "node", "test",
                                    std::vector<NodePtr>{in1, in2, in3});
  };

  virtual void TearDown(){};

};


struct testElemwiseEmitter : public Emitter {
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<OpNode> node;

  NgraphNodePtr data1;
  NgraphNodePtr data2;

  testElemwiseEmitter(nnvmNodePtr n) {
    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");

    node = std::make_shared<OpNode>(n, "node", "test",
                                    std::vector<NodePtr>{in1, in2});

    in1->shape_ = nnvm::TShape{2, 4, 8};
    in2->shape_ = nnvm::TShape{2, 4, 8};
    node->shape_ = nnvm::TShape{2, 4, 8};

    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in2->shape_));

    data1 = op_map_[in1];
    data2 = op_map_[in2];
  };
};

struct testEmitterBroadcast : public Emitter {
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<OpNode> node;
  NgraphNodePtr data1;
  NgraphNodePtr data2;
  testEmitterBroadcast() {
    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");
    node = std::make_shared<OpNode>(nullptr, "node", "test",
                                    std::vector<NodePtr>{in1, in2});

    in1->shape_ = nnvm::TShape{2, 3, 1, 5};
    in2->shape_ = nnvm::TShape{2, 1, 4, 5};
    node->shape_ = nnvm::TShape{2, 3, 4, 5};

    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in2->shape_));
    data1 = op_map_[in1];
    data2 = op_map_[in2];
  };
};

struct testBatchNormEmitter : public Emitter, public ::testing::Test {
 protected:
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<VariableNode> in3;
  std::shared_ptr<VariableNode> in4;
  std::shared_ptr<VariableNode> in5;
  std::shared_ptr<OpNode> node;

  NgraphNodePtr data1;
  NgraphNodePtr data2;
  NgraphNodePtr data3;
  NgraphNodePtr data4;
  NgraphNodePtr data5;

  virtual void SetUp() {
      in1 = std::make_shared<VariableNode>(nullptr, "in1");
      in2 = std::make_shared<VariableNode>(nullptr, "in2");
      in3 = std::make_shared<VariableNode>(nullptr, "in3");
      in4 = std::make_shared<VariableNode>(nullptr, "in4");
      in5 = std::make_shared<VariableNode>(nullptr, "in5");

      node = std::make_shared<OpNode>(nullptr, "node", "test",
                                      std::vector<NodePtr>{in1, in2, in3, in4, in5});
  };

  virtual void TearDown(){};

};

}  // namespace ngraph_bridge
