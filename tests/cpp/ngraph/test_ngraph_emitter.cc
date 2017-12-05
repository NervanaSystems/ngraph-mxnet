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

#include "test_ngraph_emitter.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/operator/concat-inl.h"
#include "../../src/operator/fully_connected-inl.h"
#include "../../src/operator/slice_channel-inl.h"
#include "../../src/operator/tensor/matrix_op-inl.h"

namespace ngraph_bridge {

testElemwiseEmitter test_emitter(nullptr);

TEST(NGRAPH_EMITTER, COMPOUND_UNARY_OPS) {
  auto relu = test_emitter.ngraph_op_funcs_["relu"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(relu));

  EXPECT_EQ(relu->get_arguments()[0], test_emitter.data1);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Constant>(
      relu->get_arguments()[1]));
  EXPECT_EQ(
      std::dynamic_pointer_cast<ngraph::op::Constant>(relu->get_arguments()[1])
          ->get_value_strings()[0],
      "0");

  auto sigmoid = test_emitter.ngraph_op_funcs_["sigmoid"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(sigmoid));

  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Add>(sigmoid->get_arguments()[1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Exp>(
      sigmoid->get_arguments()[1]->get_arguments()[1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Negative>(
      sigmoid->get_arguments()[1]->get_arguments()[1]->get_arguments()[0]));
  EXPECT_EQ(sigmoid->get_arguments()[1]
                ->get_arguments()[1]
                ->get_arguments()[0]
                ->get_arguments()[0],
            test_emitter.data1);

  EXPECT_EQ(test_emitter.ngraph_op_funcs_["_copy"](test_emitter.node),
            test_emitter.data1);

  auto recip = test_emitter.ngraph_op_funcs_["reciprocal"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(recip));
  EXPECT_EQ(
      std::dynamic_pointer_cast<ngraph::op::Constant>(recip->get_arguments()[0])
          ->get_value_strings()[0],
      "1");
  EXPECT_EQ(recip->get_arguments()[1], test_emitter.data1);

  auto square = test_emitter.ngraph_op_funcs_["square"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(square));
  EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                square->get_arguments()[1])
                ->get_value_strings()[0],
            "2");

  auto sqrt = test_emitter.ngraph_op_funcs_["sqrt"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(sqrt));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Divide>(sqrt->get_arguments()[1]));
  EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                sqrt->get_arguments()[1]->get_arguments()[1])
                ->get_value_strings()[0],
            "2");

  auto rsqrt = test_emitter.ngraph_op_funcs_["rsqrt"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(rsqrt));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Power>(rsqrt->get_arguments()[1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
      rsqrt->get_arguments()[1]->get_arguments()[1]));
  EXPECT_EQ(
      std::dynamic_pointer_cast<ngraph::op::Constant>(
          rsqrt->get_arguments()[1]->get_arguments()[1]->get_arguments()[1])
          ->get_value_strings()[0],
      "2");

  auto cbrt = test_emitter.ngraph_op_funcs_["cbrt"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(cbrt));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Divide>(cbrt->get_arguments()[1]));
  EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                cbrt->get_arguments()[1]->get_arguments()[1])
                ->get_value_strings()[0],
            "3");

  auto rcbrt = test_emitter.ngraph_op_funcs_["rcbrt"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(rcbrt));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Power>(rcbrt->get_arguments()[1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
      rcbrt->get_arguments()[1]->get_arguments()[1]));
  EXPECT_EQ(
      std::dynamic_pointer_cast<ngraph::op::Constant>(
          rcbrt->get_arguments()[1]->get_arguments()[1]->get_arguments()[1])
          ->get_value_strings()[0],
      "3");

  auto log2 = test_emitter.ngraph_op_funcs_["log2"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(log2));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Log>(log2->get_arguments()[0]));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Log>(log2->get_arguments()[1]));
  EXPECT_EQ(log2->get_arguments()[0]->get_arguments()[0], test_emitter.data1);
  EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                log2->get_arguments()[1]->get_arguments()[0])
                ->get_value_strings()[0],
            "2");

  auto log10 = test_emitter.ngraph_op_funcs_["log10"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(log10));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Log>(log10->get_arguments()[0]));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Log>(log10->get_arguments()[1]));
  EXPECT_EQ(log10->get_arguments()[0]->get_arguments()[0], test_emitter.data1);
  EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                log10->get_arguments()[1]->get_arguments()[0])
                ->get_value_strings()[0],
            "10");

  auto degrees = test_emitter.ngraph_op_funcs_["degrees"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(degrees));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
      degrees->get_arguments()[1]));

  auto radians = test_emitter.ngraph_op_funcs_["radians"](test_emitter.node);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(radians));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
      radians->get_arguments()[1]));
}

TEST(NGRAPH_EMITTER, SIMPLE_UNARY_OPS) {
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Negative>(
      test_emitter.ngraph_op_funcs_["negative"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Abs>(
      test_emitter.ngraph_op_funcs_["abs"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Ceiling>(
      test_emitter.ngraph_op_funcs_["ceil"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Floor>(
      test_emitter.ngraph_op_funcs_["floor"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Exp>(
      test_emitter.ngraph_op_funcs_["exp"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Log>(
      test_emitter.ngraph_op_funcs_["log"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sin>(
      test_emitter.ngraph_op_funcs_["sin"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cos>(
      test_emitter.ngraph_op_funcs_["cos"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tan>(
      test_emitter.ngraph_op_funcs_["tan"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Asin>(
      test_emitter.ngraph_op_funcs_["arcsin"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Acos>(
      test_emitter.ngraph_op_funcs_["arccos"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Atan>(
      test_emitter.ngraph_op_funcs_["arctan"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sinh>(
      test_emitter.ngraph_op_funcs_["sinh"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cosh>(
      test_emitter.ngraph_op_funcs_["cosh"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tanh>(
      test_emitter.ngraph_op_funcs_["tanh"](test_emitter.node)));
}

TEST(NGRAPH_EMITTER, BINARY_OPS) {
  // elementwise ops
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test_emitter.ngraph_op_funcs_["_plus"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Subtract>(
      test_emitter.ngraph_op_funcs_["_minus"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(
      test_emitter.ngraph_op_funcs_["_mul"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
      test_emitter.ngraph_op_funcs_["_div"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Remainder>(
      test_emitter.ngraph_op_funcs_["_mod"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
      test_emitter.ngraph_op_funcs_["_power"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(
      test_emitter.ngraph_op_funcs_["_maximum"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Minimum>(
      test_emitter.ngraph_op_funcs_["_minimum"](test_emitter.node)));

  // Logic
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Equal>(
      test_emitter.ngraph_op_funcs_["_equal"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::NotEqual>(
      test_emitter.ngraph_op_funcs_["_not_equal"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Greater>(
      test_emitter.ngraph_op_funcs_["_greater"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::GreaterEq>(
      test_emitter.ngraph_op_funcs_["_greater_equal"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Less>(
      test_emitter.ngraph_op_funcs_["_lesser"](test_emitter.node)));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::LessEq>(
      test_emitter.ngraph_op_funcs_["_lesser_equal"](test_emitter.node)));
}

TEST_F(testGeneralEmitter, DOT) {
  in1->shape_ = nnvm::TShape{2, 4};
  in2->shape_ = nnvm::TShape{4, 2};
  node->shape_ = nnvm::TShape{2, 2};

  op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), TShape_to_NShape(in1->shape_));
  op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), TShape_to_NShape(in2->shape_));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Dot>(
      ngraph_op_funcs_["dot"](node)));
}

TEST_F(testGeneralEmitter, SPLIT) {
  // slice no squeeze
  {
    mxnet::op::SliceChannelParam param;
    param.num_outputs = 4;
    param.axis = 2;
    param.squeeze_axis = 0;

    nnvm::NodeAttrs attr;
    attr.name = "split_no_squeeze";
    attr.op = (nnvm::Op*)mxnet::op::CreateOp<mxnet::cpu>(param, 0);
    attr.dict["num_outputs"] = "4";
    attr.dict["axis"] = "2";
    attr.dict["squeeze_axis"] = "0";

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1, in2, in3});

    in1->shape_ = nnvm::TShape{2, 4, 8, 16};
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    node->shape_ = nnvm::TShape{2, 4, 2, 16};
    node->multi_output_index_ = 1;

    auto op = ngraph_op_funcs_["split"](node);

    ASSERT_TRUE(std::dynamic_pointer_cast<ngraph::op::Slice>(op));

    auto op_cast = std::dynamic_pointer_cast<ngraph::op::Slice>(op);
    EXPECT_EQ(op_cast->get_lower_bounds(), ngraph::Shape({0, 0, 2, 0}));
    EXPECT_EQ(op_cast->get_upper_bounds(), ngraph::Shape({2, 4, 4, 16}));
    EXPECT_EQ(op_cast->get_step(), ngraph::Shape({1, 1, 1, 1}));
    EXPECT_EQ(op_cast->get_shape(), TShape_to_NShape(node->shape_));
  }
  // slice with squeeze
  {
    mxnet::op::SliceChannelParam param;
    param.num_outputs = 8;
    param.axis = 2;
    param.squeeze_axis = 1;

    nnvm::NodeAttrs attr;
    attr.name = "split_no_squeeze";
    attr.op = (nnvm::Op*)mxnet::op::CreateOp<mxnet::cpu>(param, 0);
    attr.dict["num_outputs"] = "8";
    attr.dict["axis"] = "2";
    attr.dict["squeeze_axis"] = "1";

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1, in2, in3});

    in1->shape_ = nnvm::TShape{2, 4, 8, 16};
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    node->shape_ = nnvm::TShape{2, 4, 16};
    node->multi_output_index_ = 0;

    auto op = ngraph_op_funcs_["split"](node);

    ASSERT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>(op));

    auto op_cast = std::dynamic_pointer_cast<ngraph::op::Reshape>(op);

    ngraph::AxisVector order(TShape_to_NShape(in1->shape_).size());
    std::iota(order.begin(), order.end(), 0);

    EXPECT_EQ(op_cast->get_input_order(), order);
    EXPECT_EQ(op_cast->get_shape(), TShape_to_NShape(node->shape_));
  }
}

TEST_F(testGeneralEmitter, CONCAT) {
  // concat
  {
    mxnet::op::ConcatParam param;
    param.num_args = 2;
    param.dim = 0;

    nnvm::NodeAttrs attr;
    attr.name = "concat";
    attr.dict["num_args"] = "2";
    attr.dict["dim"] = "0";
    auto in1shape = nnvm::TShape{2, 2, 2};
    auto in2shape = nnvm::TShape{2, 2, 2};
    auto svec = std::vector<nnvm::TShape>{in1shape, in2shape};
    attr.op = (nnvm::Op*)mxnet::op::CreateOp<mxnet::cpu>(param, 0, &svec);

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1, in2});

    in1->shape_ = in1shape;
    in2->shape_ = in2shape;
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in2->shape_));
    node->shape_ = nnvm::TShape{4, 2, 2};
    std::cout << "make concat" << std::endl;
    auto op = std::dynamic_pointer_cast<ngraph::op::Concat>(
        ngraph_op_funcs_["concat"](node));
    ASSERT_TRUE(op);
    EXPECT_EQ(op->get_concatenation_axis(), 0);
    EXPECT_EQ(op->get_shape(), TShape_to_NShape(node->shape_));
  }
}

TEST(NGRAPH_EMITTER, BROADCAST_OPS) {
  testEmitterBroadcast test_broadcast;
  auto test_direct_op = [&test_broadcast](std::string opname) {
    auto op = test_broadcast.ngraph_op_funcs_[opname](test_broadcast.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Broadcast>(
        op->get_arguments()[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Broadcast>(
        op->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>(
        (op->get_arguments()[0])->get_arguments()[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>(
        (op->get_arguments()[1])->get_arguments()[0]));
    return op;
  };

  auto op = test_direct_op("broadcast_add");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(op));
  op = test_direct_op("broadcast_sub");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Subtract>(op));
  op = test_direct_op("broadcast_mul");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(op));
  op = test_direct_op("broadcast_div");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(op));
  op = test_direct_op("broadcast_mod");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Remainder>(op));
  op = test_direct_op("broadcast_power");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(op));
  op = test_direct_op("broadcast_maximum");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(op));
  op = test_direct_op("broadcast_minimum");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Minimum>(op));
  op = test_direct_op("broadcast_equal");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Equal>(op));
  op = test_direct_op("broadcast_not_equal");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::NotEqual>(op));
  op = test_direct_op("broadcast_greater");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Greater>(op));
  op = test_direct_op("broadcast_greater_equal");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::GreaterEq>(op));
  op = test_direct_op("broadcast_lesser");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Less>(op));
  op = test_direct_op("broadcast_lesser_equal");
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::LessEq>(op));
}

TEST_F(testGeneralEmitter, MATRIX_OPS) {
  // expand dims
  {
    nnvm::NodeAttrs attr;
    attr.name = "expanddim_test";
    attr.dict["axis"] = "0";
    attr.op = nnvm::Op::Get("expand_dims");

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1, in2});

    in1->shape_ = nnvm::TShape{2, 2};
    in2->shape_ = nnvm::TShape{2, 2};
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in2->shape_));
    node->shape_ = nnvm::TShape{1, 2, 2};

    auto op = std::dynamic_pointer_cast<ngraph::op::Reshape>(
        ngraph_op_funcs_["expand_dims"](node));
    ASSERT_TRUE(op);
    EXPECT_EQ(op->get_input_order(), ngraph::Shape({0, 1}));
    EXPECT_EQ(op->get_shape(), TShape_to_NShape(node->shape_));
  }
  // flatten
  {
    nnvm::NodeAttrs attr;
    attr.name = "flatten_test";
    attr.op = nnvm::Op::Get("flatten");

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1});

    in1->shape_ = nnvm::TShape{2, 4, 8, 16};
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    node->shape_ = nnvm::TShape{2, 4 * 8 * 16};

    auto op = std::dynamic_pointer_cast<ngraph::op::Reshape>(
        ngraph_op_funcs_["flatten"](node));
    ASSERT_TRUE(op);
    EXPECT_EQ(op->get_input_order(), ngraph::Shape({0, 1, 2, 3}));
    EXPECT_EQ(op->get_shape(), TShape_to_NShape(node->shape_));
  }
  // transpose
  {
    nnvm::NodeAttrs attr;
    attr.name = "transpose_test";
    attr.op = nnvm::Op::Get("transpose");

    auto nnvmnode = nnvm::Node::Create();
    nnvmnode->attrs = attr;
    node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                    std::vector<NodePtr>{in1});

    in1->shape_ = nnvm::TShape{2, 4};
    op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Float32::element_type(),
        TShape_to_NShape(in1->shape_));
    node->shape_ = nnvm::TShape{4, 2};

    auto op = std::dynamic_pointer_cast<ngraph::op::Reshape>(
        ngraph_op_funcs_["transpose"](node));
    ASSERT_TRUE(op);
    EXPECT_EQ(op->get_input_order(), ngraph::Shape({1, 0}));
    EXPECT_EQ(op->get_shape(), TShape_to_NShape(node->shape_));
  }
}

TEST_F(testGeneralEmitter, FULLYCONNECTED) {
  mxnet::op::FullyConnectedParam param;
  param.num_hidden = 8;

  nnvm::NodeAttrs attr;
  attr.name = "concat";
  attr.dict["num_hidden"] = "8";
  auto in1shape = nnvm::TShape{2, 4};
  auto nodeshape = nnvm::TShape{2, 8};
  auto inshape = std::vector<nnvm::TShape>{in1shape};
  auto outshape = std::vector<nnvm::TShape>{nodeshape};
  attr.op = (nnvm::Op*)mxnet::op::CreateOp<mxnet::cpu>(
      param, 0, &inshape, &outshape, mxnet::Context());

  auto nnvmnode = nnvm::Node::Create();
  nnvmnode->attrs = attr;
  node = std::make_shared<OpNode>(nnvmnode, "node", "test",
                                  std::vector<NodePtr>{in1, in2, in3});

  in1->shape_ = in1shape;
  in2->shape_ = nnvm::TShape{8, 4};
  in3->shape_ = nnvm::TShape{8};
  op_map_[in1] = std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), TShape_to_NShape(in1->shape_));
  op_map_[in2] = std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), TShape_to_NShape(in2->shape_));
  op_map_[in3] = std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), TShape_to_NShape(in3->shape_));
  node->shape_ = nodeshape;

  data1 = op_map_[in1];
  data2 = op_map_[in2];
  data3 = op_map_[in3];

  auto op = ngraph_op_funcs_["FullyConnected"](node);

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(op));
  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Dot>(op->get_arguments()[0]));

  EXPECT_EQ(op->get_arguments()[0]->get_arguments()[0], data1);
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>(
      op->get_arguments()[0]->get_arguments()[1]));
  EXPECT_EQ(op->get_arguments()[0]->get_arguments()[1]->get_arguments()[0],
            data2);

  EXPECT_TRUE(
      std::dynamic_pointer_cast<ngraph::op::Broadcast>(op->get_arguments()[1]));
  EXPECT_EQ(op->get_arguments()[1]->get_arguments()[0], data3);
}

}  // namespace ngraph_bridge
