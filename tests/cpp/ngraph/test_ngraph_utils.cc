#include "test_util.h"
#include "../../src/ngraph/ngraph_graph_utils.h"
#include "../../src/ngraph/ngraph_compiler_utils.h"
#include "../../src/ngraph/ngraph_pycompiler_utils.h"
namespace ngraph {

struct testAxes_utils {
  testAxes_utils() {
    InitializePython();
    gil_state state;
    ng = py::module::import("ngraph");
    N = ng.attr("make_axis")("length"_a = 32, "name"_a = "N");
    C = ng.attr("make_axis")("length"_a = 3, "name"_a = "C");
    D = ng.attr("make_axis")("length"_a = 1, "name"_a = "D");
    H = ng.attr("make_axis")("length"_a = 28, "name"_a = "H");
    W = ng.attr("make_axis")("length"_a = 28, "name"_a = "W");
  }
  py::module ng;
  py::object N;
  py::object C;
  py::object D;
  py::object H;
  py::object W;
  py::object place(py::tuple axes) { return ng.attr("placeholder")(axes); }
};

testAxes_utils axes;

TEST(NGRAPH_AXES, NUM_AXES) {
  EXPECT_EQ(num_axes(axes.place(
                createPyTuple(pyvec{axes.N, axes.C, axes.D, axes.H, axes.W}))),
            5);
  EXPECT_EQ(num_axes(axes.place(
                createPyTuple(pyvec{axes.N, axes.C, axes.H, axes.W}))),
            4);
  EXPECT_EQ(num_axes(axes.place(createPyTuple(pyvec{axes.N, axes.H}))), 2);
}

TEST(NGRAPH_AXES, GETNTHAXIS) {
  EXPECT_TRUE(getNthAxis(axes.place(createPyTuple(
                             pyvec{axes.N, axes.C, axes.D, axes.H, axes.W})),
                         3) == axes.H);
  EXPECT_TRUE(getNthAxis(axes.place(createPyTuple(
                             pyvec{axes.N, axes.C, axes.D, axes.H, axes.W})),
                         1) == axes.C);
  EXPECT_ANY_THROW(getNthAxis(
      axes.place(createPyTuple(pyvec{axes.N, axes.C, axes.D, axes.H, axes.W})),
      5));
}

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

TEST(NGRAPH_STRING, CLEAN_OPNAME) {
  EXPECT_EQ(clean_opname("elemwise_add"), "add");
  EXPECT_EQ(clean_opname("broadcast_add"), "add");
}

}  // namespace ngraph