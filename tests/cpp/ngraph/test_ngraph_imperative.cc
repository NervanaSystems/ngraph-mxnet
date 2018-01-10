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

#include "test_ngraph_imperative.h"

namespace ngraph_bridge {

TEST_F(NGRAPH_IMPERATIVE, CREATE_IMPERATIVE) {
  testImperative test(attrs, mxnet::Context::CPU(), inputs, nullptr, outputs);
  EXPECT_FALSE(test.ngraph_.ngraph_forward);
  EXPECT_EQ(test.ngraph_.context_, mxnet::Context::CPU());
  EXPECT_EQ(test.GetInputs().size(), inputs.size());
  EXPECT_EQ(test.graph_.outputs.size(), outputs.size());
}

TEST_F(NGRAPH_IMPERATIVE, SYMBOL_GRAPH) {
  std::string test_name = "elemwise_mul";
  nnvm::NodeAttrs sym_attrs;
  sym_attrs.op = nnvm::Op::Get(test_name);
  testImperative test(sym_attrs, mxnet::Context::CPU(), inputs, nullptr,
                      outputs);
  const auto &idx = test.graph_.indexed_graph();
  EXPECT_EQ(idx.num_nodes(), inputs.size() + 1);
  EXPECT_EQ(idx[(const int &)0].source->attrs.name, test_name + "_var_0");
  EXPECT_EQ(idx[inputs.size()].source->attrs.name, test_name);
}

TEST_F(NGRAPH_IMPERATIVE, PARSE_OPGRAPH) {
  testImperative test(attrs, mxnet::Context::CPU(), inputs, nullptr, outputs);
  EXPECT_EQ(test.ngraph_.nodes_.size(), 0);
  EXPECT_FALSE(test.op_ngraph_);
  test.parse_ngraph();
  EXPECT_TRUE(test.op_ngraph_);
  EXPECT_EQ(test.ngraph_.nodes_.size(), inputs.size() + 1);
  for (auto n : test.ngraph_.nodes_) {
    if (n->type_ == NodeType::kGraph)
      EXPECT_EQ(n->in_ngraph_, true);
    else
      EXPECT_EQ(n->in_ngraph_, false);
  }
  for (auto n : test.ngraph_.nodes_) {
    if (n->type_ == NodeType::kOp)
      EXPECT_EQ(n->operation_, op_name);
    else
      EXPECT_NE(n->operation_, op_name);
  }
}

TEST_F(NGRAPH_IMPERATIVE, UNSUPPORTED_OP) {
  nnvm::NodeAttrs sym_attrs;
  sym_attrs.op = nnvm::Op::Get("IdentityAttachKLSparseReg");
  testImperative test(sym_attrs, mxnet::Context::CPU(), outputs, nullptr,
                      outputs);
  EXPECT_EQ(test.ngraph_.nodes_.size(), 0);
  test.parse_ngraph();
  EXPECT_EQ(test.ngraph_.nodes_.size(), 2);
  for (auto n : test.ngraph_.nodes_) {
    EXPECT_EQ(n->in_ngraph_, false);
  }
}

TEST_F(NGRAPH_IMPERATIVE, INVOKE_OP) {
  testImperative test(attrs, mxnet::Context::CPU(), inputs, nullptr, outputs);
  auto op_ng = test.get_op_ngraph();

  EXPECT_TRUE(op_ng);
  EXPECT_TRUE(test.op_ngraph_->ngraph_forward);

  auto placeholders = make_ngraph_placeholders(
      inputs, GetBackendFromContext(op_ng->context_), true);
  auto results = make_ngraph_placeholders(
      outputs, GetBackendFromContext(op_ng->context_), false);

  EXPECT_EQ(vec3, std::vector<float>({0, 0}));
  results.insert(results.end(), op_ng->cached_values.begin(),
                 op_ng->cached_values.end());
  op_ng->ngraph_forward->call(placeholders, results);
  result_to_TBlob(results[0], outputs, 0);
  EXPECT_EQ(vec3, std::vector<float>({2, 6}));
}

TEST_F(NGRAPH_IMPERATIVE, CHECKOPS) {
  EXPECT_TRUE(NGImperative::check_op_supported("_plus"));
  // verify alias ops
  EXPECT_TRUE(NGImperative::check_op_supported("elemwise_add"));
  // test layer ops
  EXPECT_TRUE(NGImperative::check_op_supported("Activation"));
  // test unsupported op
  EXPECT_FALSE(NGImperative::check_op_supported("IdentityAttachKLSparseReg"));
}

}  // namespace ngraph_bridge
