#include "test_ngraph_compiler.h"
namespace ngraph_bridge {

TEST_F(NGRAPH_COMPILER, DEEPCOPY) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  for (auto kv : test.nodeMap_){
    EXPECT_NE(kv.first, kv.second.get());
    EXPECT_EQ(kv.first->attrs.name, kv.second->attrs.name);
    EXPECT_EQ(kv.first->attrs.op, kv.second->attrs.op);
    ASSERT_EQ(kv.first->inputs.size(), kv.second->inputs.size());
    if (kv.first->inputs.size()>0)
      for (size_t i = 0; i < kv.first->inputs.size(); ++i) {
        EXPECT_NE(kv.first->inputs[i].node.get(),
                  kv.second->inputs[i].node.get());
        EXPECT_EQ(kv.first->inputs[i].node->attrs.name,
                  kv.second->inputs[i].node->attrs.name);
        EXPECT_EQ(kv.first->inputs[i].node->attrs.op,
                  kv.second->inputs[i].node->attrs.op);
      }
  }
}

TEST_F(NGRAPH_COMPILER, COPIED_INPUTS) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  auto out_inputs = test.GetInputs();
  int i = 0;
  for (auto n : inputs){
    EXPECT_NE(n, out_inputs[i]);
    EXPECT_EQ(test.nodeMap_[n.get()], out_inputs[i]);
    i += 1;
  }
}

TEST_F(NGRAPH_COMPILER, CLEAN_OPNAME) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);

  EXPECT_EQ(clean_opname("elemwise_add"), "_plus");
  EXPECT_EQ(clean_opname("_add"), "_plus");
  EXPECT_EQ(clean_opname("_Plus"), "_plus");
  EXPECT_EQ(clean_opname("_sub"), "_minus");
  EXPECT_EQ(clean_opname("_Minus"), "_minus");
  EXPECT_EQ(clean_opname("_Mul"), "_mul");
  EXPECT_EQ(clean_opname("_Div"), "_div");
  EXPECT_EQ(clean_opname("_Mid"), "_mod");
  EXPECT_EQ(clean_opname("_Power"),"_power");
  EXPECT_EQ(clean_opname("_Maximum"),"_maximum");
  EXPECT_EQ(clean_opname("_Minimum"), "_minimum");
  EXPECT_EQ(clean_opname("_Hypot"),"_hypot");
  EXPECT_EQ(clean_opname("_Equal"), "_equal");
  EXPECT_EQ(clean_opname("_Not_Equal"), "_not_equal");
  EXPECT_EQ(clean_opname("_Greater"), "_greater");
  EXPECT_EQ(clean_opname("_Greater_Equal"), "_greater_equal");
  EXPECT_EQ(clean_opname("_Lesser"), "_lesser");
  EXPECT_EQ(clean_opname("_Lesser_Equal"), "_lesser_equal");
}

TEST_F(NGRAPH_COMPILER, PARSENNVMGRAPH){
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  EXPECT_FALSE(true);
}

}