#include "test_util.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler.h"

namespace ngraph_bridge {

  class testSGCompiler : public SGCompiler {
  public:
    NodePtr in1;
    NodePtr in2;
    NodePtr in3;
    NodePtr node1;
    NodePtr node2;
    std::shared_ptr<Graph> subgraph;
    using SGCompiler::op_map;
    using SGCompiler::NgraphOpFuncs_;
    using SGCompiler::CompileInput;
    using SGCompiler::CompileNode;
    using SGCompiler::Compile;


    testSGCompiler() {
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
    };

    std::shared_ptr<ngraph::Node> operator[](NodePtr node){
      return op_map[node];
    }

    int count(NodePtr node){
      return op_map.count(node);
    }

  };

  TEST(NGRAPH_SGCOMPILER, COMPILE_PARAMETER){
    testSGCompiler test;
    EXPECT_FALSE(test.count(test.in1));
    test.CompileInput(test.in1);
    EXPECT_TRUE(test.count(test.in1));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in1]));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.in1]->get_value_type())
                    ->get_shape(),
                TShape_to_NShape(test.in1->shape));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.in1]->get_value_type())
                    ->get_element_type(),
                getType(test.in1->dtype));
  }

  TEST(NGRAPH_SGCOMPILER, COMPILE_NODE1){
    testSGCompiler test;
    EXPECT_FALSE(test.count(test.in1));
    EXPECT_FALSE(test.count(test.in2));
    EXPECT_FALSE(test.count(test.in3));
    EXPECT_FALSE(test.count(test.node1));
    EXPECT_FALSE(test.count(test.node2));
    test.CompileNode(test.node1, test.subgraph);

    EXPECT_TRUE(test.count(test.in1));
    EXPECT_TRUE(test.count(test.in2));
    EXPECT_FALSE(test.count(test.in3));
    EXPECT_TRUE(test.count(test.node1));
    EXPECT_FALSE(test.count(test.node2));


    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in2]));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test[test.node1]));
  }

  TEST(NGRAPH_SGCOMPILER, COMPILE_NODE2){
    testSGCompiler test;
    EXPECT_FALSE(test.count(test.in1));
    EXPECT_FALSE(test.count(test.in2));
    EXPECT_FALSE(test.count(test.in3));
    EXPECT_FALSE(test.count(test.node1));
    EXPECT_FALSE(test.count(test.node2));

    test.CompileNode(test.node2, test.subgraph);

    EXPECT_TRUE(test.count(test.in1));
    EXPECT_TRUE(test.count(test.in2));
    EXPECT_TRUE(test.count(test.in3));
    EXPECT_TRUE(test.count(test.node1));
    EXPECT_TRUE(test.count(test.node2));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in2]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in3]));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test[test.node1]));
  }
  
  TEST(NGRAPH_SGCOMPILER, COMPILE_SUBGRAPH){
    testSGCompiler test;
    EXPECT_FALSE(test.subgraph->ngraph_forward);
    EXPECT_FALSE(test.subgraph->ngraph_backward);
    EXPECT_FALSE(test.count(test.in1));
    EXPECT_FALSE(test.count(test.in2));
    EXPECT_FALSE(test.count(test.in3));
    EXPECT_FALSE(test.count(test.node1));
    EXPECT_FALSE(test.count(test.node2));

    test.Compile(test.subgraph);

    EXPECT_TRUE(test.count(test.in1));
    EXPECT_TRUE(test.count(test.in2));
    EXPECT_TRUE(test.count(test.in3));
    EXPECT_TRUE(test.count(test.node1));
    EXPECT_TRUE(test.count(test.node2));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in2]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Parameter>(
        test[test.in3]));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test[test.node1]));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.node1]->get_value_type())
                    ->get_shape(),
                TShape_to_NShape(test.node1->shape));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.node1]->get_value_type())
                    ->get_element_type(),
                getType(test.node1->dtype));

    EXPECT_TRUE(std::dynamic_pointer_cast<ngraph::op::Add>(
        test[test.node2]));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.node2]->get_value_type())
                    ->get_shape(),
                TShape_to_NShape(test.node2->shape));
    EXPECT_EQ(std::dynamic_pointer_cast<const ngraph::TensorViewType>(
                    test[test.node2]->get_value_type())
                    ->get_element_type(),
                getType(test.node2->dtype));
    EXPECT_TRUE(test.subgraph->ngraph_forward);
    // EXPECT_TRUE(test.subgraph->ngraph_backward); //Not yet Implemented
  }
}

