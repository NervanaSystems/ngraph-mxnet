#include "test_util.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler.h"

namespace ngraph_bridge {


class NGRAPH_SGCOMPILER : public ::testing::Test {
protected:
  NodePtr in1;
  NodePtr in2;
  NodePtr in3;
  NodePtr node1;
  NodePtr node2;
  std::shared_ptr<Graph> subgraph;
  virtual void SetUp() {
    const auto shape = nnvm::TShape{4,8,12,16};

    in1 = std::make_shared<VariableNode>(nullptr, "in1");
    in2 = std::make_shared<VariableNode>(nullptr, "in2");
    in3 = std::make_shared<VariableNode>(nullptr, "in3");

    node1 = std::make_shared<OpNode>(nullptr, "node1", "_plus",
                                     std::vector<NodePtr>{in1, in2});
    node2 = std::make_shared<OpNode>(nullptr, "node2", "_plus",
                                     std::vector<NodePtr>{node1, in3});

    in1->shape = shape;
    in2->shape = shape;
    in3->shape = shape;
    node1->shape = shape;
    node2->shape = shape;

    subgraph = std::make_shared<Graph>();
    subgraph->inputs.push_back(in1);
    subgraph->inputs.push_back(in2);
    subgraph->inputs.push_back(in3);
    subgraph->nodes_.push_back(node1);
    subgraph->nodes_.push_back(node2);
  }

  virtual void TearDown(){};
};

class testSGCompiler : public SGCompiler {
  public:
    using SGCompiler::op_map;
    using SGCompiler::NgraphOpFuncs_;
    using SGCompiler::CompileInput;
    using SGCompiler::CompileNode;
    using SGCompiler::Compile;

    std::shared_ptr<ngraph::Node> operator[](NodePtr node){
      return op_map[node];
    }

    int count(NodePtr node){
      return op_map.count(node);
    }

  };

TEST_F(NGRAPH_SGCOMPILER, COMPILE_PARAMETER){
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  test.CompileInput(in1);
  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in1]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[in1]->get_value_type())
                  ->get_shape(),
              TShape_to_NShape(in1->shape));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[in1]->get_value_type())
                  ->get_element_type(),
              getType(in1->dtype));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_NODE1){
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));
  test.CompileNode(node1, subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_FALSE(test.count(node2));


  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in2]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test[node1]));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_NODE2){
  testSGCompiler test;
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));

  test.CompileNode(node2, subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_TRUE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_TRUE(test.count(node2));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in2]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in3]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test[node1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test[node2]));
}

TEST_F(NGRAPH_SGCOMPILER, COMPILE_SUBGRAPH){
  testSGCompiler test;
  EXPECT_FALSE(subgraph->ngraph_forward);
  EXPECT_FALSE(subgraph->ngraph_backward);
  EXPECT_FALSE(test.count(in1));
  EXPECT_FALSE(test.count(in2));
  EXPECT_FALSE(test.count(in3));
  EXPECT_FALSE(test.count(node1));
  EXPECT_FALSE(test.count(node2));

  test.Compile(subgraph);

  EXPECT_TRUE(test.count(in1));
  EXPECT_TRUE(test.count(in2));
  EXPECT_TRUE(test.count(in3));
  EXPECT_TRUE(test.count(node1));
  EXPECT_TRUE(test.count(node2));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in2]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
      test[in3]));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test[node1]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[node1]->get_value_type())
                  ->get_shape(),
              TShape_to_NShape(node1->shape));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[node1]->get_value_type())
                  ->get_element_type(),
              getType(node1->dtype));

  EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
      test[node2]));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[node2]->get_value_type())
                  ->get_shape(),
              TShape_to_NShape(node2->shape));
  EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                  test[node2]->get_value_type())
                  ->get_element_type(),
              getType(node2->dtype));
  EXPECT_TRUE(subgraph->ngraph_forward);
  // EXPECT_TRUE(subgraph->ngraph_backward); //Not yet Implemented
}

}

