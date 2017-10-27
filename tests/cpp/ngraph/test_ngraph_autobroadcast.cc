#include "test_util.h"
#include "../../src/ngraph/ngraph_autobroadcast.h"

namespace ngraph_bridge {

std::shared_ptr<ngraph::op::Parameter> getParamFromShape(
    const ngraph::Shape &shape) {
  return std::make_shared<ngraph::op::Parameter>(
      ngraph::element::Float32::element_type(), shape);
}

void propTypes(const NgraphNodePtr &node) {
  for (auto arg : node->get_arguments()) {
    propTypes(arg);
  }
  node->propagate_types();
}

inline ngraph::Shape getShapeFromParam(const NgraphNodePtr &node) {
  propTypes(node);
  auto type = std::dynamic_pointer_cast<const ngraph::TensorViewType>(
      node->get_value_type());
  return type->get_shape();
}

// input shapes are equal so AutoBroadcast does nothing
TEST(NGRAPH_AUTOBROADCAST, NO_BROADCAST_EQUAL) {
  ngraph::Shape s2345{2, 3, 4, 5};
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s2345);
  AutoBroadcast ab(lhs, s2345, rhs, s2345);

  EXPECT_EQ(ab.lhs(), lhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  EXPECT_EQ(ab.rhs(), rhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

// input shapes are incompatable so AutoBroadcast does nothing
TEST(NGRAPH_AUTOBROADCAST, NO_BROADCAST_INCOMPATABLE) {
  ngraph::Shape s2345{2, 3, 4, 5};
  ngraph::Shape s6789{6, 7, 8, 9};
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s6789);
  AutoBroadcast ab(lhs, s2345, rhs, s6789);

  EXPECT_EQ(ab.lhs(), lhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  EXPECT_EQ(ab.rhs(), rhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s6789);
}

// basic broadcast test
// 1D to 2D
// lhs broadcast to 2,3
TEST(NGRAPH_AUTOBROADCAST, NORMAL_BROADCAST_2D) {
  ngraph::Shape s3{3};
  ngraph::Shape s23{2, 3};
  auto lhs = getParamFromShape(s3);
  auto rhs = getParamFromShape(s23);
  AutoBroadcast ab(lhs, s3, rhs, s23);

  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s23);

  EXPECT_EQ(ab.rhs(), rhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s23);
}

// basic broadcast test
// 2D to 3D
// lhs broadcast to 2,3,4
TEST(NGRAPH_AUTOBROADCAST, NORMAL_BROADCAST_3D) {
  ngraph::Shape s34{3, 4};
  ngraph::Shape s234{2, 3, 4};
  auto lhs = getParamFromShape(s34);
  auto rhs = getParamFromShape(s234);
  AutoBroadcast ab(lhs, s34, rhs, s234);

  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s234);

  EXPECT_EQ(ab.rhs(), rhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s234);
}

// basic broadcast test
// 3D to 4D
// lhs broadcast to 2,3,4,5
TEST(NGRAPH_AUTOBROADCAST, NORMAL_BROADCAST_4D) {
  ngraph::Shape s345{3, 4, 5};
  ngraph::Shape s2345{2, 3, 4, 5};
  auto lhs = getParamFromShape(s345);
  auto rhs = getParamFromShape(s2345);
  AutoBroadcast ab(lhs, s345, rhs, s2345);

  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  EXPECT_EQ(ab.rhs(), rhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

// basic reshape and broadcast test
// rhs reshape to 2,3,4 then
// rhs broadcast to 2,3,4,5
TEST(NGRAPH_AUTOBROADCAST, RESHAPE_1X_BROADCAST) {
  ngraph::Shape s2345{2, 3, 4, 5};
  ngraph::Shape s2341{2, 3, 4, 1};
  auto lhs = getParamFromShape(s2345);
  auto rhs = getParamFromShape(s2341);
  AutoBroadcast ab(lhs, s2345, rhs, s2341);

  EXPECT_EQ(ab.lhs(), lhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

// same as above, but additionally
// lhs reshape to 2,4,5 then
// lhs broadcast to 2,3,4,5
TEST(NGRAPH_AUTOBROADCAST, RESHAPE_2X_BROADCAST) {
  ngraph::Shape s2145{2, 1, 4, 5};
  ngraph::Shape s2341{2, 3, 4, 1};
  auto lhs = getParamFromShape(s2145);
  auto rhs = getParamFromShape(s2341);
  AutoBroadcast ab(lhs, s2145, rhs, s2341);

  ngraph::Shape s2345{2, 3, 4, 5};

  EXPECT_NE(ab.lhs(), lhs);
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2345);

  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2345);
}

// matching singular dimension on axis 2
// should not require reshape of either lhs or rhs
// i.e. this should be the same as normal broadcast casse
// rhs broadcast to 2,3,1,5
TEST(NGRAPH_AUTOBROADCAST, BROADCAST_WITH_DIM1) {
  ngraph::Shape s2315{2, 3, 1, 5};
  ngraph::Shape s315{3, 1, 5};
  auto lhs = getParamFromShape(s2315);
  auto rhs = getParamFromShape(s315);
  AutoBroadcast ab(lhs, s2315, rhs, s315);

  EXPECT_EQ(ab.lhs(), lhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s2315);

  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s2315);
}

// reshape only test
// rhs reshape to 1,3,4,5 with no broadcast
TEST(NGRAPH_AUTOBROADCAST, BROADCAST_WITH_LEADING_DIM1) {
  ngraph::Shape s1345{1, 3, 4, 5};
  ngraph::Shape s345{3, 4, 5};
  auto lhs = getParamFromShape(s1345);
  auto rhs = getParamFromShape(s345);
  AutoBroadcast ab(lhs, s1345, rhs, s345);

  EXPECT_EQ(ab.lhs(), lhs);  // no change
  EXPECT_EQ(getShapeFromParam(ab.lhs()), s1345);

  EXPECT_NE(ab.rhs(), rhs);
  EXPECT_EQ(getShapeFromParam(ab.rhs()), s1345);
}

// handle edge input case - nullptr
TEST(NGRAPH_AUTOBROADCAST, NULL_POINTER) {
  ngraph::Shape s345{3, 4, 5};
  auto param = getParamFromShape(s345);
  std::shared_ptr<ngraph::op::Parameter> nullparam = nullptr;
  EXPECT_THROW(AutoBroadcast ab(nullparam, s345, param, s345), const char *);
  EXPECT_THROW(AutoBroadcast ab(param, s345, nullparam, s345), const char *);
}

// handle edge input case - empty shape
TEST(NGRAPH_AUTOBROADCAST, EMPTY_SHAPE) {
  ngraph::Shape sEmpty{};
  ngraph::Shape s345{3, 4, 5};
  auto param = getParamFromShape(s345);
  EXPECT_THROW(AutoBroadcast ab(param, sEmpty, param, s345), const char *);
  EXPECT_THROW(AutoBroadcast ab(param, s345, param, sEmpty), const char *);
}

// handle edge input case - invalid shape (zero dimension)
TEST(NGRAPH_AUTOBROADCAST, ZERO_DIMENSION) {
  ngraph::Shape s305{3, 0, 5};
  ngraph::Shape s345{3, 4, 5};
  auto param = getParamFromShape(s345);
  EXPECT_THROW(AutoBroadcast ab(param, s305, param, s345), const char *);
  EXPECT_THROW(AutoBroadcast ab(param, s345, param, s305), const char *);
}

}  // namespace ngraph_bridge
