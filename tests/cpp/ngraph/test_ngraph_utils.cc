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

TEST(NGRAPH_SGCOMPILER, Convert_Shapes){
  auto Tshape = nnvm::TShape{2,3,4,5};
  auto Nshape = TShape_to_NShape(Tshape);

  std::vector<int> TshapeVec;
  std::vector<int> NshapeVec;
  for (auto t : Tshape) TshapeVec.push_back(t);
  for (auto n : Nshape) NshapeVec.push_back(n);
  for (int i = 0; i<4; ++i) EXPECT_EQ(TshapeVec[i], NshapeVec[i]);
}

}  // namespace ngraph