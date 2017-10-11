#include "test_util.h"
#include "ngraph_compiler.h"
#include <nnvm/graph.h>


class CompilerTest: public ::testing::Test {

protected:
virtual void SetUp(){
//The idea here is to create a graph for (A*B) dot product
//This has 2 inputs and one output
//Initialize two input nodes
nnvm::NodeAttrs attr;
in1 = nnvm::Node::Create();
in2 = nnvm::Node::Create();
op_node = nnvm::Node::Create();
attr.name = "in1";
attr.name = "in1";
attr.name = "in1";
in1->attrs = attr;
attr.name = "in2";
in2->attrs = attr;
//Create op node
const nnvm::Op* temp_op = nnvm::Op::Get("dot");
attr.op = temp_op;

op_node->attrs = attr;

nnvm::NodeEntry ne0;
nnvm::NodeEntry ne1;
nnvm::NodeEntry ne2;

ne0.node = in1;
ne1.node = in2;
op_node->inputs.push_back(ne0);
op_node->inputs.push_back(ne1);
ne2.node = op_node; 
nnvm_graph.outputs.push_back(ne2);

//Adding the necessary boilerplate code to get the Compiler object created
nnvm::TShape shape({2,2});
std::vector<int> dtypes({1,1,1});
std::vector<nnvm::TShape> shapes({shape, shape, shape});
nnvm_graph.attrs["shape"] = std::make_shared<nnvm::any>(std::move(shapes)); 
nnvm_graph.attrs["dtype"] = std::make_shared<nnvm::any>(std::move(dtypes));
};

virtual void TearDown(){};

nnvm::NodePtr in1; 
nnvm::NodePtr in2; 
nnvm::NodePtr op_node;
nnvm::Graph nnvm_graph;
ngraph_bridge::NDArrayMap feed_dict;
ngraph_bridge::NNVMNodeVec inputs;
int temp = 1;


};



