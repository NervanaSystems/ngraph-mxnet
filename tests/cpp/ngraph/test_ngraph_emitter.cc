#include "test_ngraph_emitter.h"

namespace ngraph_bridge {

  testEmitter test_emitter;
  testEmitterBroadcast test_broadcast;

  TEST(NGRAPH_EMITTER, COMPOUND_UNARY_OPS) {
    auto relu = test_emitter.NgraphOpFuncs_["relu"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(relu));

    EXPECT_EQ(relu->get_arguments()[0], test_emitter.data1);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Constant>(
        relu->get_arguments()[1]));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  relu->get_arguments()[1])
                  ->get_value_strings()[0],
              "0");

    auto sigmoid = test_emitter.NgraphOpFuncs_["sigmoid"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(sigmoid));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        sigmoid->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Exp>(
        sigmoid->get_arguments()[1]->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Negative>(
        sigmoid->get_arguments()[1]->get_arguments()[1]->get_arguments()[0]));
    EXPECT_EQ(sigmoid->get_arguments()[1]
                  ->get_arguments()[1]
                  ->get_arguments()[0]
                  ->get_arguments()[0],
              test_emitter.data1);

    EXPECT_EQ(test_emitter.NgraphOpFuncs_["_copy"](test_emitter.node),
              test_emitter.data1);

    auto recip = test_emitter.NgraphOpFuncs_["reciprocal"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(recip));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  recip->get_arguments()[0])
                  ->get_value_strings()[0],
              "1");
    EXPECT_EQ(recip->get_arguments()[1], test_emitter.data1);

    auto square = test_emitter.NgraphOpFuncs_["square"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(square));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  square->get_arguments()[1])
                  ->get_value_strings()[0],
              "2");

    auto sqrt = test_emitter.NgraphOpFuncs_["sqrt"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(sqrt));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        sqrt->get_arguments()[1]));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  sqrt->get_arguments()[1]->get_arguments()[1])
                  ->get_value_strings()[0],
              "2");

    auto rsqrt = test_emitter.NgraphOpFuncs_["rsqrt"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(rsqrt));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        rsqrt->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        rsqrt->get_arguments()[1]->get_arguments()[1]));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  rsqrt->get_arguments()[1]->get_arguments()[1]->get_arguments()[1])
                  ->get_value_strings()[0],
              "2");

    auto cbrt = test_emitter.NgraphOpFuncs_["cbrt"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(cbrt));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        cbrt->get_arguments()[1]));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  cbrt->get_arguments()[1]->get_arguments()[1])
                  ->get_value_strings()[0],
              "3");

    auto rcbrt = test_emitter.NgraphOpFuncs_["rcbrt"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(rcbrt));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        rcbrt->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        rcbrt->get_arguments()[1]->get_arguments()[1]));
    EXPECT_EQ(
        std::dynamic_pointer_cast<ngraph::op::Constant>(
            rcbrt->get_arguments()[1]->get_arguments()[1]->get_arguments()[1])
            ->get_value_strings()[0],
        "3");

    auto log2 = test_emitter.NgraphOpFuncs_["log2"](test_emitter.node);
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

    auto log10 = test_emitter.NgraphOpFuncs_["log10"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(log10));
    EXPECT_TRUE(
        std::dynamic_pointer_cast<ngraph::op::Log>(log10->get_arguments()[0]));
    EXPECT_TRUE(
        std::dynamic_pointer_cast<ngraph::op::Log>(log10->get_arguments()[1]));
    EXPECT_EQ(log10->get_arguments()[0]->get_arguments()[0],
              test_emitter.data1);
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  log10->get_arguments()[1]->get_arguments()[0])
                  ->get_value_strings()[0],
              "10");

    auto degrees = test_emitter.NgraphOpFuncs_["degrees"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(degrees));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        degrees->get_arguments()[1]));

    auto radians = test_emitter.NgraphOpFuncs_["radians"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(radians));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        radians->get_arguments()[1]));
  }

  TEST(NGRAPH_EMITTER, SIMPLE_UNARY_OPS) {
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Negative>(
        test_emitter.NgraphOpFuncs_["negative"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Abs>(
        test_emitter.NgraphOpFuncs_["abs"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Ceiling>(
        test_emitter.NgraphOpFuncs_["ceil"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Floor>(
        test_emitter.NgraphOpFuncs_["floor"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Exp>(
        test_emitter.NgraphOpFuncs_["exp"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Log>(
        test_emitter.NgraphOpFuncs_["log"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sin>(
        test_emitter.NgraphOpFuncs_["sin"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cos>(
        test_emitter.NgraphOpFuncs_["cos"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tan>(
        test_emitter.NgraphOpFuncs_["tan"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Asin>(
        test_emitter.NgraphOpFuncs_["arcsin"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Acos>(
        test_emitter.NgraphOpFuncs_["arccos"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Atan>(
        test_emitter.NgraphOpFuncs_["arctan"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Sinh>(
        test_emitter.NgraphOpFuncs_["sinh"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Cosh>(
        test_emitter.NgraphOpFuncs_["cosh"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Tanh>(
        test_emitter.NgraphOpFuncs_["tanh"](test_emitter.node)));
  }

  TEST(NGRAPH_EMITTER, BINARY_OPS) {
    //elementwise ops
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test_emitter.NgraphOpFuncs_["_plus"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Subtract>(
        test_emitter.NgraphOpFuncs_["_minus"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Multiply>(
        test_emitter.NgraphOpFuncs_["_mul"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        test_emitter.NgraphOpFuncs_["_div"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        test_emitter.NgraphOpFuncs_["_power"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Maximum>(
        test_emitter.NgraphOpFuncs_["_maximum"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Minimum>(
        test_emitter.NgraphOpFuncs_["_minimum"](test_emitter.node)));

    auto hypot = test_emitter.NgraphOpFuncs_["_hypot"](test_emitter.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(hypot));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Divide>(
        hypot->get_arguments()[1]));
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  hypot->get_arguments()[1]->get_arguments()[0])
                  ->get_value_strings()[0],
              "1");
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  hypot->get_arguments()[1]->get_arguments()[1])
                  ->get_value_strings()[0],
              "2");
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        hypot->get_arguments()[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        hypot->get_arguments()[0]->get_arguments()[0]));
    EXPECT_EQ(hypot->get_arguments()[0]->get_arguments()[0]->get_arguments()[0],
              test_emitter.data1);
    EXPECT_EQ(std::dynamic_pointer_cast<ngraph::op::Constant>(
                  hypot->get_arguments()[0]->get_arguments()[0]->get_arguments()[1])
                  ->get_value_strings()[0],
              "2");
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Power>(
        hypot->get_arguments()[0]->get_arguments()[1]));

    EXPECT_EQ(hypot->get_arguments()[0]->get_arguments()[1]->get_arguments()[0],
              test_emitter.data2);
    EXPECT_EQ(
        std::dynamic_pointer_cast<ngraph::op::Constant>(
            hypot->get_arguments()[0]->get_arguments()[1]->get_arguments()[1])
            ->get_value_strings()[0],
        "2");

    //Logic
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Equal>(
        test_emitter.NgraphOpFuncs_["_equal"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::NotEqual>(
        test_emitter.NgraphOpFuncs_["_not_equal"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Greater>(
        test_emitter.NgraphOpFuncs_["_greater"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::GreaterEq>(
        test_emitter.NgraphOpFuncs_["_greater_equal"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Less>(
        test_emitter.NgraphOpFuncs_["_lesser"](test_emitter.node)));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::LessEq>(
        test_emitter.NgraphOpFuncs_["_lesser_equal"](test_emitter.node)));
    //other
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Dot>(
        test_emitter.NgraphOpFuncs_["dot"](test_emitter.node)));

    // broadcast
    auto broadcast_add = test_broadcast.NgraphOpFuncs_["broadcast_add"](test_broadcast.node);
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(broadcast_add));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Broadcast>(broadcast_add->get_arguments()[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Broadcast>(broadcast_add->get_arguments()[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>((broadcast_add->get_arguments()[0])->get_arguments()[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Reshape>((broadcast_add->get_arguments()[1])->get_arguments()[0]));
  }

}

