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

#include "test_util.h"

#include "../../src/ngraph/ngraph_graph_utils.h"
#include "../../src/ngraph/ngraph_nnvm_utils.h"
#include "../../src/ngraph/ngraph_sgcompiler_utils.h"
namespace ngraph_bridge {

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

TEST(NGRAPH_SGCOMPILER_UTILS, Convert_Shapes) {
  auto Tshape = nnvm::TShape{2, 3, 4, 5};
  auto Nshape = TShape_to_NShape(Tshape);

  std::vector<int> TshapeVec;
  std::vector<int> NshapeVec;

  for (auto t : Tshape) TshapeVec.push_back(t);
  for (auto n : Nshape) NshapeVec.push_back(n);

  for (int i = 0; i < 4; ++i) EXPECT_EQ(TshapeVec[i], NshapeVec[i]);
}

TEST(NGRAPH_SGCOMPILER_UTILS, GetNGraphTypes) {
  EXPECT_EQ(ngraph::element::Float32::element_type(),
            getType(mshadow::kFloat32));
  EXPECT_EQ(ngraph::element::UInt8::element_type(), getType(mshadow::kUint8));
  EXPECT_EQ(ngraph::element::Int8::element_type(), getType(mshadow::kInt8));
  EXPECT_EQ(ngraph::element::Int32::element_type(), getType(mshadow::kInt32));
  EXPECT_EQ(ngraph::element::Int64::element_type(), getType(mshadow::kInt64));
}

TEST(NGRAPH_NNVM, GetBufferSize) {
  std::vector<int> vecshape{2, 3, 4, 5};
  ngraph::Shape ngshape{2, 3, 4, 5};
  nnvm::TShape Tshape{2, 3, 4, 5};

  EXPECT_EQ(get_buffer_size(vecshape, 2), 240);
  EXPECT_EQ(get_buffer_size(vecshape, 4), 480);
  EXPECT_EQ(get_buffer_size(vecshape, 8), 960);
  EXPECT_EQ(get_buffer_size(ngshape, 2), 240);
  EXPECT_EQ(get_buffer_size(ngshape, 4), 480);
  EXPECT_EQ(get_buffer_size(ngshape, 8), 960);
  EXPECT_EQ(get_buffer_size(Tshape, 2), 240);
  EXPECT_EQ(get_buffer_size(Tshape, 4), 480);
  EXPECT_EQ(get_buffer_size(Tshape, 8), 960);
}

TEST(NGRAPH_NNVM, CopyBuffers) {
  auto manager = ngraph::runtime::Manager::get("NGVM");
  auto backend = manager->allocate_backend();
  ngraph::Shape shape{10};
  // TODO: Is there a shorter way to write this?
  {
    auto PT = backend->make_parameterized_tensor_view<ngraph::element::Float32>(
        shape);
    std::vector<float> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    copy_TBlob(
        PT, mshadow::kFloat32,
        get_buffer_size(shape, ngraph::element::Float32::element_type().size()),
        vec.data());

    std::vector<float> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
    copy_result(
        PT, mshadow::kFloat32,
        get_buffer_size(shape, ngraph::element::Float32::element_type().size()),
        vec2.data());
    EXPECT_EQ(vec2, PT->get_vector());
  }
  {
    auto PT =
        backend->make_parameterized_tensor_view<ngraph::element::UInt8>(shape);
    std::vector<uint8_t> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    copy_TBlob(
        PT, mshadow::kUint8,
        get_buffer_size(shape, ngraph::element::UInt8::element_type().size()),
        vec.data());

    std::vector<uint8_t> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
    copy_result(
        PT, mshadow::kUint8,
        get_buffer_size(shape, ngraph::element::UInt8::element_type().size()),
        vec2.data());
    EXPECT_EQ(vec2, PT->get_vector());
  }
  {
    auto PT =
        backend->make_parameterized_tensor_view<ngraph::element::Int8>(shape);
    std::vector<int8_t> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    copy_TBlob(
        PT, mshadow::kInt8,
        get_buffer_size(shape, ngraph::element::Int8::element_type().size()),
        vec.data());

    std::vector<int8_t> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
    copy_result(
        PT, mshadow::kInt8,
        get_buffer_size(shape, ngraph::element::Int8::element_type().size()),
        vec2.data());
    EXPECT_EQ(vec2, PT->get_vector());
  }
  {
    auto PT =
        backend->make_parameterized_tensor_view<ngraph::element::Int32>(shape);
    std::vector<int32_t> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    copy_TBlob(
        PT, mshadow::kInt32,
        get_buffer_size(shape, ngraph::element::Int32::element_type().size()),
        vec.data());

    std::vector<int32_t> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
    copy_result(
        PT, mshadow::kInt32,
        get_buffer_size(shape, ngraph::element::Int32::element_type().size()),
        vec2.data());
    EXPECT_EQ(vec2, PT->get_vector());
  }
  {
    auto PT =
        backend->make_parameterized_tensor_view<ngraph::element::Int64>(shape);
    std::vector<int64_t> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    copy_TBlob(
        PT, mshadow::kInt64,
        get_buffer_size(shape, ngraph::element::Int64::element_type().size()),
        vec.data());

    std::vector<int64_t> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};
    copy_result(
        PT, mshadow::kInt64,
        get_buffer_size(shape, ngraph::element::Int64::element_type().size()),
        vec2.data());
    EXPECT_EQ(vec2, PT->get_vector());
  }
}

TEST(NGRAPH_NNVM, copy_TBlobs) {
  nnvm::TShape shape{10};
  std::vector<float> vec1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<float> vec2{11, 12, 13, 14, 15, 16, 17, 18, 19, 10};

  mxnet::TBlob TBlob1(vec1.data(), shape, 0);
  mxnet::TBlob TBlob2(vec2.data(), shape, 0);
  std::vector<mxnet::TBlob> inblobs;
  inblobs.push_back(TBlob1);
  inblobs.push_back(TBlob2);

  auto placeholders = make_ngraph_placeholders(inblobs, true);

  EXPECT_EQ(
      vec1,
      std::dynamic_pointer_cast<
          ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>(
          placeholders[0])
          ->get_vector());
  EXPECT_EQ(
      vec2,
      std::dynamic_pointer_cast<
          ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>(
          placeholders[1])
          ->get_vector());
  std::vector<float> vec3{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> vec4{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  mxnet::TBlob TBlob3(vec3.data(), shape, 0);
  mxnet::TBlob TBlob4(vec4.data(), shape, 0);
  std::vector<mxnet::TBlob> outblobs;
  outblobs.push_back(TBlob3);
  outblobs.push_back(TBlob4);

  result_to_TBlob(placeholders[0], outblobs, 0);
  result_to_TBlob(placeholders[1], outblobs, 1);
  EXPECT_EQ(vec1, vec3);
  EXPECT_EQ(vec2, vec4);
}

}  // namespace ngraph_bridge
