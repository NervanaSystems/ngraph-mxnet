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

#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

using ValueVector = std::vector<std::shared_ptr<ngraph::runtime::Value>>;
using ngraph::runtime::ParameterizedTensorView;
using ngraph::runtime::TensorView;

// This function copies the data in an ngraph tensor into a void*
// (that presumably came from a TBlob)
template <typename ET>
inline void copy_TBlob(std::shared_ptr<ngraph::runtime::Value> input, size_t n,
                       void* p) {
  auto PT = std::dynamic_pointer_cast<ParameterizedTensorView<ET>>(input);
  PT->write(p, 0, n);
}
// Runtime type identification and switching of compile-time templated
// functions to execute copy_TBlob properly
// TODO: The switch statements aren't very good. is there a better way?
inline void copy_TBlob(std::shared_ptr<ngraph::runtime::Value> input,
                       int type_flag, size_t n, void* p) {
  switch (type_flag) {
    // case mshadow::kFloat16:
    //   copy_result<ngraph::element::Float16>(input, n, p);
    case mshadow::kFloat32:
      copy_TBlob<ngraph::element::Float32>(input, n, p);
      break;
    // case mshadow::kFloat64:
    //   copy_TBlob<ngraph::element::Float64>(input, n, p);
    case mshadow::kUint8:
      copy_TBlob<ngraph::element::UInt8>(input, n, p);
      break;
    case mshadow::kInt8:
      copy_TBlob<ngraph::element::Int8>(input, n, p);
      break;
    case mshadow::kInt32:
      copy_TBlob<ngraph::element::Int32>(input, n, p);
      break;
    case mshadow::kInt64:
      copy_TBlob<ngraph::element::Int64>(input, n, p);
      break;
    default:
      throw "NGRAPH_BRIDGE: type not supported";
  }
}

// This function copies the data in a void*
// (that presumably came from a TBlob) to an ngraph tensor
template <typename ET>
inline void copy_result(std::shared_ptr<ngraph::runtime::Value> input, size_t n,
                        void* p) {
  auto PT = std::dynamic_pointer_cast<ParameterizedTensorView<ET>>(input);
  PT->read(p, 0, n);
}

// Runtime type identification and switching of compile-time templated
// functions to execute copy_result properly
inline void copy_result(std::shared_ptr<ngraph::runtime::Value> input,
                        int type_flag, size_t n, void* p) {
  switch (type_flag) {
    // case mshadow::kFloat16:
    //   copy_result<ngraph::element::Float16>(input, n, p);
    case mshadow::kFloat32:
      copy_result<ngraph::element::Float32>(input, n, p);
      break;
    // case mshadow::kFloat64:
    //   copy_result<ngraph::element::Float64>(input, n, p);
    case mshadow::kUint8:
      copy_result<ngraph::element::UInt8>(input, n, p);
      break;
    case mshadow::kInt8:
      copy_result<ngraph::element::Int8>(input, n, p);
      break;
    case mshadow::kInt32:
      copy_result<ngraph::element::Int32>(input, n, p);
      break;
    case mshadow::kInt64:
      copy_result<ngraph::element::Int64>(input, n, p);
      break;
    default:
      throw "NGRAPH_BRIDGE: type not supported";
  }
}

// Simple utility for getting the total number of bytes in a
// buffer, either from an mxnet tensor or an ngraph tensor
// TODO: Would std::accumulate work for mxnet TShapes?
template <typename T>
inline size_t get_buffer_size(const T& shape, size_t nbytes) {
  size_t out = nbytes;
  for (const auto& i : shape) out *= i;
  return out;
}

// This function creates an ngraph Tensor from the shape and type
// of an input mxnet TBlob. It optionally copies the data
// from the TBlob to the ngraph tensor.
inline std::shared_ptr<TensorView> TBlob_to_TensorView(
    const mxnet::TBlob& input, bool copy = false) {
  auto shape = TShape_to_NShape(input.shape_);
  const auto& element_type = getType(input.type_flag_);

  auto TV = element_type.make_primary_tensor_view(shape);

  if (copy) {
    auto buffer_size = get_buffer_size(shape, element_type.size());
    copy_TBlob(TV, input.type_flag_, buffer_size, input.dptr_);
  }

  return TV;
}

// Main utility funciton for creating NNVM ops
// This function takes a vector of TBlobs and creates a vector of
// equialently shaped and typed ngraph tensors, optionally
// copied the data from the TBlobs to ngraph
// TODO:: std::transform?
inline ValueVector make_ngraph_placeholders(
    const std::vector<mxnet::TBlob>& inputs, bool copy_data = false) {
  ValueVector out;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto TV = TBlob_to_TensorView(inputs[i], copy_data);
    out.emplace_back(TV);
  }

  return out;
}

// Utility function that copies the outnum'th result from an
// ngraph computation into the outnum'th output TBlob in mxnet
// TODO: Make this loop over the outputs to copy all results at once?
template <typename T>
inline void result_to_TBlob(T& result, const std::vector<mxnet::TBlob>& outputs,
                            int outnum) {
  void* p = outputs[outnum].dptr_;
  const auto& element_type = getType(outputs[outnum].type_flag_);
  auto buffer_size =
      get_buffer_size(outputs[outnum].shape_, element_type.size());
  copy_result(result, outputs[outnum].type_flag_, buffer_size,
              outputs[outnum].dptr_);
}

}  // namespace ngraph_bridge
