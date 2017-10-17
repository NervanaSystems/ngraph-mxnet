#include "test_ngraph_compiler.h"
namespace ngraph_bridge {

TEST_F(NGRAPH_COMPILER, DEEPCOPY){
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

}