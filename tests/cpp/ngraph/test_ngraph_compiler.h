#include "test_util.h"
#include "ngraph_compiler.h"
#include <nnvm/graph.h>

class CompilerTest: public ::testing::Test {

protected:
virtual void SetUp(){
//The idea here is to create a graph for (A+B)
//Initialize two input nodes
nnvm::NodeAttrs attr;
attr.name = "in1";
in1->attrs = attr;
attr.name = "in2";
in2->attrs = attr;
//Create op node
std::shared_ptr<nnvm::Op> temp_op;
temp_op->name = "add";
temp_op->set_num_inputs(2);
temp_op->set_num_outputs(1);
attr.op = temp_op.get();
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

};

virtual void TearDown(){};

nnvm::NodePtr in1;
nnvm::NodePtr in2;
nnvm::NodePtr op_node;
nnvm::Graph nnvm_graph;
int temp = 1;


};



