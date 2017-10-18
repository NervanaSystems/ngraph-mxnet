#include <nnvm/graph.h>
#include "ngraph_compiler.h"
#include "test_util.h"

namespace ngraph_bridge{

class NGRAPH_COMPILER : public ::testing::Test {
 protected:
  nnvm::NodeEntry createNode(std::string name, std::string op = "") {
    nnvm::NodeAttrs attr;
    auto node = nnvm::Node::Create();
    attr.name = name;
    if (op != "") attr.op = nnvm::Op::Get(op);
    node->attrs = attr;
    return nnvm::NodeEntry{node, 0, 0};
  }
  

  virtual void SetUp() {
    auto A = createNode("A");
    auto B = createNode("B");
    auto C = createNode("C");
    auto D = createNode("D");
    auto add1 = createNode("add1", "_add");
    auto mul = createNode("mul", "_mul");
    auto add2 = createNode("add2", "_add");
    auto relu = createNode("relu", "relu");

    add1.node->inputs.push_back(A);
    add1.node->inputs.push_back(B);

    mul.node->inputs.push_back(add1);
    mul.node->inputs.push_back(C);

    add2.node->inputs.push_back(mul);
    add2.node->inputs.push_back(D);

    relu.node->inputs.push_back(add2);

    nnvm_graph.outputs.push_back(relu);

    nnvm::TShape shape{2, 2};
    std::unordered_map<std::string, int> dtypes;
    std::unordered_map<std::string, nnvm::TShape> shapes;

    for (auto n : {A, B, C, D}) inputs.push_back(n.node);

    for (auto n : {"A", "B", "C", "D"}) {
      dtypes[n] = 0;
      shapes[n] = shape;
    }

    bindarg = std::make_shared<ngraph_bridge::SimpleBindArg>(4, shapes, dtypes);
  };

  virtual void TearDown(){};

  nnvm::Graph nnvm_graph;
  std::shared_ptr<ngraph_bridge::SimpleBindArg> bindarg;

  NDArrayMap feed_dict;
  NNVMNodeVec inputs;
};

class testCompiler : public Compiler{
 public:
  using Compiler::CheckInNGraph;
  using Compiler::DeepCopy;
  using Compiler::CopyNodes;
  using Compiler::makeCopiedFeedDict;
  using Compiler::makeCopiedInputs;
  using Compiler::Infer;
  using Compiler::nodeMap_;
  using Compiler::graph_;
  testCompiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
               const NNVMNodeVec& inputs, const BindArgBase& bindarg)
      : Compiler(graph, feed_dict, inputs, bindarg){};
};

}