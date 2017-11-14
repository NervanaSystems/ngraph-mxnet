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

#include "test_ngraph_sgcompiler.h"

namespace ngraph_bridge {

TEST_F(NGRAPH_SGCOMPILER, COMPILE_PARAMETER) {
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  test.CompileInput(in1);
  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in1]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[in1]->get_value_type())
                ->get_shape(),
            TShape_to_NShape(in1->shape_));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[in1]->get_value_type())
                ->get_element_type(),
            getType(in1->dtype_));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_NODE1) {
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));
  test.CompileNode(node1, subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_FALSE(test.count(node2));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in2]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(test[node1]));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_NODE2) {
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));

  test.CompileNode(node2, subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_TRUE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_TRUE(test.count(node2));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in2]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in3]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(test[node1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(test[node2]));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_SUBGRAPH) {
  testSGCompiler test;
  EXPECT_FALSE(subgraph->ngraph_forward);
  EXPECT_FALSE(subgraph->ngraph_backward);
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));

  test.Compile(subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_TRUE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_TRUE(test.count(node2));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in2]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(test[in3]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(test[node1]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[node1]->get_value_type())
                ->get_shape(),
            TShape_to_NShape(node1->shape_));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[node1]->get_value_type())
                ->get_element_type(),
            getType(node1->dtype_));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(test[node2]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[node2]->get_value_type())
                ->get_shape(),
            TShape_to_NShape(node2->shape_));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                test[node2]->get_value_type())
                ->get_element_type(),
            getType(node2->dtype_));
  EXPECT_TRUE(subgraph->ngraph_forward);
  EXPECT_TRUE(subgraph->ngraph_backward);
}

}  // namespace ngraph_bridge
