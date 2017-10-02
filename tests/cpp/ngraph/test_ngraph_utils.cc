#include "test_util.h"
#include "../../src/ngraph/ngraph_graph_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
namespace ngraph_bridge {

TEST(NGRAPH_STRING, GETINTS) {
  EXPECT_EQ(getInts("(1, 2, 3)"), std::vector<int>({1, 2, 3}));
  EXPECT_EQ(getInts("(1,2,3)"), std::vector<int>({1, 2, 3}));
  EXPECT_EQ(getInts("(1, 2,3, 9,12, 17)"),
            std::vector<int>({1, 2, 3, 9, 12, 17}));
}
TEST(NGRAPH_STRING, RANDOMSTRING) {
  EXPECT_EQ(randomString(12).size(), 12);
  EXPECT_EQ(randomString(4).size(), 4);
  EXPECT_EQ(randomString(77).size(), 77);
}

}  // namespace ngraph