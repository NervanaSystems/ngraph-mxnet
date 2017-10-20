#include "test_util.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/ngraph/ngraph_autobroadcast.h"

namespace ngraph_bridge {

std::shared_ptr<ngraph::op::Parameter> getParamFromShape(const ngraph::Shape &shape) {
  return std::make_shared<ngraph::op::Parameter>(getType(0), shape);
}

void propTypes(const NgraphNodePtr &node) {
  for (auto arg : node->get_arguments()) {
    propTypes(arg);
  }
  node->propagate_types();
}

inline ngraph::Shape getShapeFromParam(const NgraphNodePtr &node) {
  propTypes(node);
  auto type = std::dynamic_pointer_cast<const ngraph::TensorViewType>(node->get_value_type());
  return type->get_shape();
}

TEST(NGRAPH_AUTOBROADCAST, NO_BROADCAST_EQUAL) {
  ngraph::Shape s2345{ 2,3,4,5 };
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s2345);
  AutoBroadcast ab(lhs, s2345, rhs, s2345);

  // lhs - check no broadcast
  EXPECT_EQ(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  // rhs - check no broadcast
  EXPECT_EQ(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

TEST(NGRAPH_AUTOBROADCAST, NO_BROADCAST_INCOMPATABLE) {
  ngraph::Shape s2345{ 2,3,4,5 };
  ngraph::Shape s6789{ 6,7,8,9 };
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s6789);
  AutoBroadcast ab(lhs, s2345, rhs, s6789);

  // lhs - check no broadcast
  EXPECT_EQ(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);
  
  // rhs - check no broadcast
  EXPECT_EQ(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s6789);
}

TEST(NGRAPH_AUTOBROADCAST, NORMAL_BROADCAST) {
  ngraph::Shape s345{ 3,4,5 };
  ngraph::Shape s2345{ 2,3,4,5 };
  auto lhs = getParamFromShape(s345);
  auto rhs = getParamFromShape(s2345);
  AutoBroadcast ab(lhs, s345, rhs, s2345);

  // lhs - check broadcast
  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  // rhs - check no broadcast
  EXPECT_EQ(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

TEST(NGRAPH_AUTOBROADCAST, RESHAPE_1X_BROADCAST) {
  ngraph::Shape s2345{ 2,3,4,5 };
  ngraph::Shape s2341{ 2,3,4,1 };
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s2341);
  AutoBroadcast ab(lhs, s2345, rhs, s2341);

  // lhs - check no broadcast
  EXPECT_EQ(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);
 
  // rhs - check broadcast
  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

TEST(NGRAPH_AUTOBROADCAST, RESHAPE_2X_BROADCAST) {
  ngraph::Shape s2145{ 2,1,4,5 };
  ngraph::Shape s2341{ 2,3,4,1 };
  auto lhs = getParamFromShape(s2145);
  auto rhs = getParamFromShape(s2341);
  AutoBroadcast ab(lhs, s2145, rhs, s2341);

  ngraph::Shape s2345{ 2,3,4,5 };

  // lhs - check broadcast
  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  // rhs - check broadcast
  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

TEST(NGRAPH_AUTOBROADCAST, BROADCAST_WITH_DIM1) {
  ngraph::Shape s2315{ 2,3,1,5 };
  ngraph::Shape s315{ 3,1,5 };
  auto lhs = getParamFromShape(s2315);
  auto rhs = getParamFromShape(s315);
  AutoBroadcast ab(lhs, s2315, rhs, s315);

  // lhs - check no broadcast
  EXPECT_EQ(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2315);

  // rhs - check broadcast
  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2315);
}

TEST(NGRAPH_AUTOBROADCAST, BROADCAST_WITH_LEADING_DIM1) {
  ngraph::Shape s1345{ 1,3,4,5 };
  ngraph::Shape s345{ 3,4,5 };
  auto lhs = getParamFromShape(s1345);
  auto rhs = getParamFromShape(s345);
  AutoBroadcast ab(lhs, s1345, rhs, s345);

  // lhs - check no broadcast
  EXPECT_EQ(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s1345);

  // rhs - check broadcast
  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s1345);
}


}  // namespace ngraph_bridge
