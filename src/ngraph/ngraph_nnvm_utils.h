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

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

using TensorViewVector =
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>>;
using ngraph::runtime::TensorView;

// Simple utility for getting the total number of bytes in a
// buffer, either from an mxnet tensor or an ngraph tensor
template <typename T>
inline size_t get_buffer_size(const T& shape, size_t nbytes) {
  return std::accumulate(shape.begin(), shape.end(), nbytes,
                         std::multiplies<size_t>());
}

// This function creates an ngraph Tensor from the shape and type
// of an input mxnet TBlob. It optionally copies the data
// from the TBlob to the ngraph tensor.
inline std::shared_ptr<TensorView> TBlob_to_TensorView(
    const mxnet::TBlob& input,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy) {
  auto shape = TShape_to_NShape(input.shape_);
  const auto& element_type = getType(input.type_flag_);

  auto TV = backend->make_primary_tensor_view(element_type, shape);

  if (copy) {
    auto buffer_size = get_buffer_size(shape, element_type.size());
    TV->write(input.dptr_, 0, buffer_size);
  }

  return TV;
}

// Main utility funciton for creating NNVM ops
// This function takes a vector of TBlobs and creates a vector of
// equialently shaped and typed ngraph tensors, optionally
// copied the data from the TBlobs to ngraph
inline TensorViewVector make_ngraph_placeholders(
    const std::vector<mxnet::TBlob>& inputs,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy_data) {
  TensorViewVector out;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(out),
                 [copy_data, backend](const mxnet::TBlob& tb) {
                   return TBlob_to_TensorView(tb, backend, copy_data);
                 });
  return out;
}

// Utility function that copies the outnum'th result from an
// ngraph computation into the outnum'th output TBlob in mxnet
// TODO(mbrookhart): Make this loop over the outputs to copy all results at
// once?
template <typename T>
inline void result_to_TBlob(const T& result,
                            const std::vector<mxnet::OpReqType>& req,
                            const std::vector<mxnet::TBlob>& outputs,
                            int outnum) {
  const auto& element_type = getType(outputs[outnum].type_flag_);
  auto buffer_size =
      get_buffer_size(outputs[outnum].shape_, element_type.size());

  void* temp = malloc(buffer_size);
  result->read(temp, 0, buffer_size);

  void* p = outputs[outnum].dptr_;
  if (req[outnum] == mxnet::kAddTo) {
    for (size_t i = 0; i < (buffer_size / element_type.size()); ++i) {

      if(element_type == ngraph::element::f32)
        *(((float*)p) + i) += *(((float*)temp) + i);
      else if(element_type == ngraph::element::f64) 
        *(((double*)p) + i) += *(((double*)temp) + i);
      else if(element_type == ngraph::element::i8)
        *(((int8_t*)p) + i) += *(((int8_t*)temp) + i);
      else if (element_type == ngraph::element::i16)
        *(((int16_t*)p) + i) += *(((int16_t*)temp) + i);
      else if (element_type == ngraph::element::i32)
        *(((int32_t*)p) + i) += *(((int32_t*)temp) + i);
      else if (element_type == ngraph::element::i64)
        *(((int64_t*)p) + i) += *(((int64_t*)temp) + i);
      else if (element_type == ngraph::element::u8)
        *(((uint8_t*)p) + i) += *(((uint8_t*)temp) + i);

    }
  } else {
    memcpy(p, temp, buffer_size);
  }

  free(temp);
}
}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
