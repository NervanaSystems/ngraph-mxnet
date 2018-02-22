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

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_

#include <mxnet/op_attr_types.h>

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

template <class T>
inline void result_plus_TBlob(void* mxnet_tblob, void* ngraph_tv,
                              size_t buffer_size) {
  T* mxnet_tblob_tptr = static_cast<T*>(mxnet_tblob);
  T* ngraph_tv_tptr = static_cast<T*>(ngraph_tv);
  for (size_t i = 0; i < (buffer_size / sizeof(T)); ++i) {
    *(mxnet_tblob_tptr + i) += *(ngraph_tv_tptr + i);
  }
}

// Utility function that copies all results from an
// ngraph computation into the output TBlobs in mxnet
inline void result_to_TBlob(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& results,
    const std::vector<mxnet::OpReqType>& req,
    const std::vector<mxnet::TBlob>& outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (req[i] == mxnet::kNullOp) continue;

    const auto& element_type = getType(outputs[i].type_flag_);
    auto buffer_size = get_buffer_size(outputs[i].shape_, element_type.size());

    void* mxnet_tblob = outputs[i].dptr_;
    if (req[i] == mxnet::kAddTo) {
      void* ngraph_tv = malloc(buffer_size);
      results[i]->read(ngraph_tv, 0, buffer_size);

      if (element_type == ngraph::element::f32)
        result_plus_TBlob<float>(mxnet_tblob, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::f64)
        result_plus_TBlob<double>(mxnet_tblob, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::u8)
        result_plus_TBlob<uint8_t>(mxnet_tblob, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i8)
        result_plus_TBlob<int8_t>(mxnet_tblob, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i32)
        result_plus_TBlob<int32_t>(mxnet_tblob, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i64)
        result_plus_TBlob<int64_t>(mxnet_tblob, ngraph_tv, buffer_size);

      free(ngraph_tv);
    } else {
      // TODO(adstraw): Add support for kWriteInplace
      results[i]->read(mxnet_tblob, 0, buffer_size);
    }
  }
}

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
