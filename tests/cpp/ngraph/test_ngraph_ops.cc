//#include "test_op.h"
#include "ngraph_compiler.h"
#include "test_ngraph_compiler.h"

TEST_F(CompilerTest, TempValue) {
  ngraph_bridge::Compiler compiler(nnvm_graph, feed_dict, inputs);
  // Sample to illustrate Test Fixture usage
  ASSERT_EQ(temp, 1);
}
