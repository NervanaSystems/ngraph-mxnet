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

namespace ngraph_bridge{
  
struct testEmitter : public Emitter {
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<VariableNode> in3;
  std::shared_ptr<OpNode> node;
  NgraphNodePtr data1;
  NgraphNodePtr data2;
  NgraphNodePtr data3;
  testEmitter(nnvmNodePtr n) {
      in1 = std::make_shared<VariableNode>(nullptr, "in1");
      in2 = std::make_shared<VariableNode>(nullptr, "in2");
      in3 = std::make_shared<VariableNode>(nullptr, "in3");
      node = std::make_shared<OpNode>(n, "node", "test",
                                      std::vector<NodePtr>{in1, in2, in3});

      op_map_[in1] = std::make_shared<ngraph::op::Parameter>();
      op_map_[in2] = std::make_shared<ngraph::op::Parameter>();
      op_map_[in3] = std::make_shared<ngraph::op::Parameter>();
      data1 = op_map_[in1];
      data2 = op_map_[in2];
      data3 = op_map_[in3];
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

    auto s2345 = nnvm::TShape{ 2,3,4,5 };
    auto s2145 = nnvm::TShape{ 2,1,4,5 };
    auto s2315 = nnvm::TShape{ 2,3,1,5 };

    in1->shape_ = s2145;
    in2->shape_ = s2315;
    node->shape_ = s2345;

    op_map_[in1] = std::make_shared<ngraph::op::Parameter>();
    op_map_[in2] = std::make_shared<ngraph::op::Parameter>();
    data1 = op_map_[in1];
    data2 = op_map_[in2];
  };
};
}
