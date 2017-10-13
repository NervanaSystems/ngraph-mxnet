#include "test_util.h"
#include "../../src/ngraph/ngraph_emitter.h"

namespace ngraph_bridge {
  Emitter test_emitter;
  auto make_data = [](){
    std::vector<std::shared_ptr<ngraph::Node> > data;
    data.push_back(std::make_shared<ngraph::op::Parameter>());
    data.push_back(std::make_shared<ngraph::op::Parameter>());
    return data;
  };
  auto data = make_data();

  TEST(NGRAPH_EMITTER, UNARY_OPS) {
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Negative>(
        test_emitter.NgraphUnaryOps_["negative"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Abs>(
        test_emitter.NgraphUnaryOps_["abs"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Ceiling>(
        test_emitter.NgraphUnaryOps_["ceil"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Floor>(
        test_emitter.NgraphUnaryOps_["floor"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Exp>(
        test_emitter.NgraphUnaryOps_["exp"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Log>(
        test_emitter.NgraphUnaryOps_["log"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sin>(
        test_emitter.NgraphUnaryOps_["sin"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cos>(
        test_emitter.NgraphUnaryOps_["cos"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tan>(
        test_emitter.NgraphUnaryOps_["tan"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Asin>(
        test_emitter.NgraphUnaryOps_["arcsin"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Acos>(
        test_emitter.NgraphUnaryOps_["arccos"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Atan>(
        test_emitter.NgraphUnaryOps_["arctan"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sinh>(
        test_emitter.NgraphUnaryOps_["sinh"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cosh>(
        test_emitter.NgraphUnaryOps_["cosh"](data[0])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tanh>(
        test_emitter.NgraphUnaryOps_["tanh"](data[0])));
  }

  TEST(NGRAPH_EMITTER, BINARY_OPS) {
    //elementwise ops
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test_emitter.NgraphBinaryOps_["_plus"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Subtract>(
        test_emitter.NgraphBinaryOps_["_minus"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(
        test_emitter.NgraphBinaryOps_["_mul"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        test_emitter.NgraphBinaryOps_["_div"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        test_emitter.NgraphBinaryOps_["_power"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(
        test_emitter.NgraphBinaryOps_["_maximum"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Minimum>(
        test_emitter.NgraphBinaryOps_["_minimum"](data[0], data[1])));
    //Logic
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Equal>(
        test_emitter.NgraphBinaryOps_["_equal"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::NotEqual>(
        test_emitter.NgraphBinaryOps_["_not_equal"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Greater>(
        test_emitter.NgraphBinaryOps_["_greater"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::GreaterEq>(
        test_emitter.NgraphBinaryOps_["_greater_equal"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Less>(
        test_emitter.NgraphBinaryOps_["_lesser"](data[0], data[1])));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::LessEq>(
        test_emitter.NgraphBinaryOps_["_lesser_equal"](data[0], data[1])));
  }

}

