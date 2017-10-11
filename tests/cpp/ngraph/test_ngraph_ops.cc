//#include "test_op.h"
#include "ngraph_compiler.h"
#include "test_ngraph_compiler.h"


TEST_F(CompilerTest, TempValue ) {

ngraph_bridge::Compiler compiler(nnvm_graph,feed_dict,inputs);
ASSERT_EQ(temp,1);
}


