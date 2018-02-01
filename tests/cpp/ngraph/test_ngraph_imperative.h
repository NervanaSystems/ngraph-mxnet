// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

#ifndef TESTS_CPP_NGRAPH_TEST_NGRAPH_IMPERATIVE_H_
#define TESTS_CPP_NGRAPH_TEST_NGRAPH_IMPERATIVE_H_

#include <string>
#include <vector>
#include "../../src/ngraph/ngraph_imperative.h"
#include "../../src/ngraph/ngraph_nnvm_utils.h"
#include "test_util.h"
namespace ngraph_bridge {

class NGRAPH_IMPERATIVE : public ::testing::Test {
 protected:
  NGRAPH_IMPERATIVE() {
    // default test arguments
    nnvm::TShape shape{2};
    inputs.emplace_back(vec1.data(), shape, 0);
    inputs.emplace_back(vec2.data(), shape, 0);
    outputs.emplace_back(vec3.data(), shape, 0);
    attrs.op = nnvm::Op::Get(op_name);
  }
  std::string op_name = "broadcast_mul";
  nnvm::NodeAttrs attrs;
  std::vector<float> vec1{1, 2};
  std::vector<float> vec2{2, 3};
  std::vector<float> vec3{0, 0};
  std::vector<mxnet::TBlob> inputs;
  std::vector<mxnet::TBlob> outputs;
};

class testImperative : public NGImperative {
 public:
  using NGImperative::graph_;
  using NGImperative::ngraph_;
  using NGImperative::op_ngraph_;
  using NGImperative::parse_ngraph;
  testImperative(const nnvm::NodeAttrs &attrs, const mxnet::Context &ctx,
                 const std::vector<mxnet::TBlob> &inputs,
                 const std::vector<mxnet::OpReqType> *req,
                 const std::vector<mxnet::TBlob> &outputs)
      : NGImperative(attrs, ctx, inputs, req, outputs) {}
};
}  // namespace ngraph_bridge
#endif  // TESTS_CPP_NGRAPH_TEST_NGRAPH_IMPERATIVE_H_
