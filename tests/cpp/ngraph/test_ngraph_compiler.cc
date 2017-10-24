// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "test_ngraph_compiler.h"
namespace ngraph_bridge {

TEST_F(NGRAPH_COMPILER, DEEPCOPY) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  for (auto kv : test.nodeMap_){
    EXPECT_NE(kv.first, kv.second.get());
    EXPECT_EQ(kv.first->attrs.name, kv.second->attrs.name);
    EXPECT_EQ(kv.first->attrs.op, kv.second->attrs.op);
    ASSERT_EQ(kv.first->inputs.size(), kv.second->inputs.size());
    if (kv.first->inputs.size()>0)
      for (size_t i = 0; i < kv.first->inputs.size(); ++i) {
        EXPECT_NE(kv.first->inputs[i].node.get(),
                  kv.second->inputs[i].node.get());
        EXPECT_EQ(kv.first->inputs[i].node->attrs.name,
                  kv.second->inputs[i].node->attrs.name);
        EXPECT_EQ(kv.first->inputs[i].node->attrs.op,
                  kv.second->inputs[i].node->attrs.op);
      }
  }
}

TEST_F(NGRAPH_COMPILER, COPIED_INPUTS) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  auto out_inputs = test.GetInputs();
  int i = 0;
  for (auto n : inputs){
    EXPECT_NE(n, out_inputs[i]);
    EXPECT_EQ(test.nodeMap_[n.get()], out_inputs[i]);
    i += 1;
  }
}

TEST_F(NGRAPH_COMPILER, COPIED_FEED_DICT) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  const auto& idx = test.graph_.indexed_graph();

  auto out_feed_dict = test.GetFeedDict();
  for (auto kv : feed_dict){
    EXPECT_NE(kv.first.node, test.nodeMap_[kv.first.node.get()]);
    // I can't find a way to compare equality of ndarry objects
    // so we're skipping this for now
    // EXPECT_EQ(out_feed_dict[kv.first], kv.second);
  }
}

TEST_F(NGRAPH_COMPILER, CLEAN_OPNAME) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  EXPECT_EQ(clean_opname("elemwise_add"), "_plus");
  EXPECT_EQ(clean_opname("elemwise_add"), "_plus");
  EXPECT_EQ(clean_opname("elemwise_sub"), "_minus");
  EXPECT_EQ(clean_opname("elemwise_mul"), "_mul");
  EXPECT_EQ(clean_opname("elemwise_div"), "_div");
  EXPECT_EQ(clean_opname("broadcast_plus"), "broadcast_add");
  EXPECT_EQ(clean_opname("broadcast_minus"), "broadcast_sub");
  EXPECT_EQ(clean_opname("_add"), "_plus");
  EXPECT_EQ(clean_opname("_Plus"), "_plus");
  EXPECT_EQ(clean_opname("_sub"), "_minus");
  EXPECT_EQ(clean_opname("_Minus"), "_minus");
  EXPECT_EQ(clean_opname("_Mul"), "_mul");
  EXPECT_EQ(clean_opname("_Div"), "_div");
  EXPECT_EQ(clean_opname("_Mod"), "_mod");
  EXPECT_EQ(clean_opname("_Power"),"_power");
  EXPECT_EQ(clean_opname("_Maximum"),"_maximum");
  EXPECT_EQ(clean_opname("_Minimum"), "_minimum");
  EXPECT_EQ(clean_opname("_Hypot"),"_hypot");
  EXPECT_EQ(clean_opname("_Equal"), "_equal");
  EXPECT_EQ(clean_opname("_Not_Equal"), "_not_equal");
  EXPECT_EQ(clean_opname("_Greater"), "_greater");
  EXPECT_EQ(clean_opname("_Greater_Equal"), "_greater_equal");
  EXPECT_EQ(clean_opname("_Lesser"), "_lesser");
  EXPECT_EQ(clean_opname("_Lesser_Equal"), "_lesser_equal");
  EXPECT_EQ(clean_opname("Concat"), "concat");
  EXPECT_EQ(clean_opname("Flatten"), "flatten");
}

TEST_F(NGRAPH_COMPILER, PARSENNVMGRAPH){
  // I haven't figured out how to create an Activation or Batchnorm Operation
  // In core NNVM. This test misses the conversion of Activation->relu
  // and the parsing of mutable nodes right now.
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  for (auto n : test.ngraph_.nodes_){
    EXPECT_EQ(n->orig_node, test.nodeMap_[nodes_[n->name].get()]);
    EXPECT_EQ(n->name, nodes_[n->name]->attrs.name);
    if (n->type == NodeType::kOp)
      EXPECT_EQ(n->operation, clean_opname(nodes_[n->name]->op()->name));
    int c = 0;
    for (auto i : n->inputs){
      EXPECT_EQ(i->orig_node, test.nodeMap_[nodes_[n->name]->inputs[c].node.get()]);
      c += 1;
    }
  }

  const auto& idx = test.graph_.indexed_graph();
  const auto inferred_shapes =
      test.graph_.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = test.graph_.GetAttr<std::vector<int>>("dtype");
  for (auto node : test.ngraph_.nodes_) {
    const uint32_t nid = idx.node_id(node->orig_node.get());
    const uint32_t eid = idx.entry_id(nid, 0);
    EXPECT_EQ(node->shape, inferred_shapes[eid]);
    EXPECT_EQ(node->dtype, inferred_dtypes[eid]);
  }
}

TEST_F(NGRAPH_COMPILER, CHECK_IN_NGRAPH){
  // this test is ignoring the multi-output check
  // Two good reasons for this:
  // 1) ngraph++ can support multiple outputs from a node
  //    so we'll need to remove that check soon
  // 2) I can't seem to create a slice operator in base nnvm :(
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  for (auto n : test.ngraph_.nodes_) {
    if (n->type == NodeType::kOp) {
      EXPECT_EQ(n->in_ngraph,
                test.compiler_.NgraphOpFuncs_.count(n->operation));
    } else {
      EXPECT_EQ(n->in_ngraph, false);
    }
  }
}

TEST_F(NGRAPH_COMPILER, COMPILE) {
  testCompiler test(nnvm_graph, feed_dict, inputs, *bindarg);
  auto out_graph = test.Compile();
  const auto& idx = out_graph.indexed_graph();
  EXPECT_EQ(idx.num_nodes(), inputs.size()+1);
  EXPECT_TRUE(out_graph.outputs[0].node->attrs.name.find("subgraph") !=
              std::string::npos);
  EXPECT_TRUE(out_graph.outputs[0].node->op()->name.find("subgraph") !=
              std::string::npos);
}

}