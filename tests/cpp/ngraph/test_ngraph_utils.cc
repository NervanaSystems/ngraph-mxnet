/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "test_util.h"

#include "../../src/ngraph/ngraph_graph_utils.h"
#include "../../src/ngraph/ngraph_nnvm_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
#include "../../src/ngraph/ngraph_utils.h"
namespace ngraph_bridge {

TEST(NGRAPH_STRING, GETINTS) {
  EXPECT_EQ(GetIntVectorFromString<int>("(1, 2, 3)"),
            std::vector<int>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<int>("(1,2,3)"),
            std::vector<int>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<int>("(1, 2,3, 9,12, 17)"),
            std::vector<int>({1, 2, 3, 9, 12, 17}));
  EXPECT_EQ(GetIntVectorFromString<int>("[1, 2, 3]]"),
            std::vector<int>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<int>("[1,2,3)"),
            std::vector<int>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<int>("[1, 2,3, 9,12, 17)))"),
            std::vector<int>({1, 2, 3, 9, 12, 17}));

  EXPECT_EQ(GetIntVectorFromString<size_t>("(1, 2, 3)"),
            std::vector<size_t>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<size_t>("(1,2,3)"),
            std::vector<size_t>({1, 2, 3}));
  EXPECT_EQ(GetIntVectorFromString<size_t>("(1, 2,3, 9,12, 17)"),
            std::vector<size_t>({1, 2, 3, 9, 12, 17}));

  EXPECT_EQ(GetIntVectorFromString<int>("(-1, 2, 3)"),
            std::vector<int>({-1, 2, 3}));
}

TEST(NGRAPH_STRING, RANDOMSTRING) {
  EXPECT_EQ(randomString(12).size(), 12ul);
  EXPECT_EQ(randomString(4).size(), 4ul);
  EXPECT_EQ(randomString(77).size(), 77ul);
}

TEST(NGRAPH_SGCOMPILER_UTILS, Convert_Shapes) {
  auto Tshape = nnvm::TShape{2, 3, 4, 5};
  auto Nshape = TShape_to_NShape(Tshape);

  std::vector<int> TshapeVec;
  std::vector<int> NshapeVec;

  for (auto t : Tshape) TshapeVec.push_back(t);
  for (auto n : Nshape) NshapeVec.push_back(n);

  for (int i = 0; i < 4; ++i) EXPECT_EQ(TshapeVec[i], NshapeVec[i]);

  EXPECT_EQ(TShape_to_NShape(nnvm::TShape{}), ngraph::Shape{});
  EXPECT_EQ(TShape_to_NShape(nnvm::TShape{1}), ngraph::Shape{1});
  EXPECT_EQ(TShape_to_NShape(nnvm::TShape{2, 3, 4, 5, 6}),
            (ngraph::Shape{2, 3, 4, 5, 6}));
  EXPECT_THROW(TShape_to_NShape(nnvm::TShape{2, 3, 4, -1}), std::runtime_error);
}

TEST(NGRAPH_SGCOMPILER_UTILS, GetNGraphTypes) {
  EXPECT_EQ(ngraph::element::f32, getType(mshadow::kFloat32));
  EXPECT_EQ(ngraph::element::u8, getType(mshadow::kUint8));
  EXPECT_EQ(ngraph::element::i8, getType(mshadow::kInt8));
  EXPECT_EQ(ngraph::element::i32, getType(mshadow::kInt32));
  EXPECT_EQ(ngraph::element::i64, getType(mshadow::kInt64));
}

TEST(NGRAPH_NNVM, GetBufferSize) {
  std::vector<int> vecshape{2, 3, 4, 5};
  ngraph::Shape ngshape{2, 3, 4, 5};
  nnvm::TShape Tshape{2, 3, 4, 5};

  EXPECT_EQ(get_buffer_size(vecshape, 2), 240ul);
  EXPECT_EQ(get_buffer_size(vecshape, 4), 480ul);
  EXPECT_EQ(get_buffer_size(vecshape, 8), 960ul);
  EXPECT_EQ(get_buffer_size(ngshape, 2), 240ul);
  EXPECT_EQ(get_buffer_size(ngshape, 4), 480ul);
  EXPECT_EQ(get_buffer_size(ngshape, 8), 960ul);
  EXPECT_EQ(get_buffer_size(Tshape, 2), 240ul);
  EXPECT_EQ(get_buffer_size(Tshape, 4), 480ul);
  EXPECT_EQ(get_buffer_size(Tshape, 8), 960ul);
}

TEST(NGRAPH_NNVM, copy_NDArrays) {
  nnvm::TShape shape{10};
  std::vector<float> vec1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<float> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};

  mxnet::NDArray array1(mxnet::TBlob(vec1.data(), shape, 1, 0), 0);
  mxnet::NDArray array2(mxnet::TBlob(vec2.data(), shape, 1, 0), 0);
  std::vector<mxnet::NDArray> inblobs;
  inblobs.push_back(array1);
  inblobs.push_back(array2);

  auto graph = std::make_shared<Graph>(Graph());
  auto backend = GetBackendFromContext(mxnet::Context::CPU());
  auto placeholders = make_ngraph_placeholders(inblobs, backend, true);

  /* EXPECT_EQ(vec1, std::dynamic_pointer_cast<ngraph::runtime::TensorView>( */
  /*                     placeholders[0]) */
  /*                     ->get_vector<float>()); */
  /* EXPECT_EQ(vec2, std::dynamic_pointer_cast<ngraph::runtime::TensorView>( */
  /*                     placeholders[1]) */
  /*                     ->get_vector<float>()); */
  std::vector<float> vec3{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> vec4{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  mxnet::NDArray array3(mxnet::TBlob(vec3.data(), shape, 1, 0), 0);
  mxnet::NDArray array4(mxnet::TBlob(vec4.data(), shape, 1, 0), 0);
  std::vector<mxnet::NDArray> outblobs;
  outblobs.push_back(array3);
  outblobs.push_back(array4);

  // test 1: kWriteTo - vec3 = vec1
  // test 2: kAddTo - vec4 += vec2
  std::vector<mxnet::OpReqType> req{mxnet::kWriteTo, mxnet::kAddTo};
  result_to_NDArray(placeholders, req, outblobs);
  // TODO(mbrookhart): Renable this tests when we have
  // per-NDArray memory sharing information
  // EXPECT_EQ(vec1, vec3);
  std::vector<float> vec4_plus_vec2{12, 13, 14, 15, 16, 17, 18, 19, 20, 11};
  EXPECT_EQ(vec4_plus_vec2, vec4);
}

}  // namespace ngraph_bridge
