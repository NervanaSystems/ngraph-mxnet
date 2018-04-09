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

#include <mxnet/ndarray.h>
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
inline std::shared_ptr<TensorView> NDArray_to_TensorView(
    const mxnet::NDArray& input,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy) {
  auto shape = TShape_to_NShape(input.shape());
  const auto& element_type = getType(input.dtype());

  // TODO(mbrookhart): I don't think Ashok's memory sharing PR has a
  // create_tensor implementation? this will probably conflict
  auto TV = backend->create_tensor(element_type, shape);

  if (copy) {
    auto buffer_size = get_buffer_size(shape, element_type.size());
    TV->write(input.storage_handle().dptr, 0, buffer_size);
  }

  return TV;
}

// Main utility funciton for creating NNVM ops
// This function takes a vector of TBlobs and creates a vector of
// equialently shaped and typed ngraph tensors, optionally
// copied the data from the TBlobs to ngraph
inline TensorViewVector make_ngraph_placeholders(
    const std::vector<mxnet::NDArray>& inputs,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy_data) {
  TensorViewVector out;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(out),
                 [copy_data, backend](const mxnet::NDArray& input) {
                   return NDArray_to_TensorView(input, backend, copy_data);
                 });
  return out;
}

template <class T>
inline void result_plus_NDArray(void* mxnet_ptr, void* ngraph_ptr,
                                size_t buffer_size) {
  T* mxnet_ptr_tptr = static_cast<T*>(mxnet_ptr);
  T* ngraph_ptr_tptr = static_cast<T*>(ngraph_ptr);
  for (size_t i = 0; i < (buffer_size / sizeof(T)); ++i) {
    *(mxnet_ptr_tptr + i) += *(ngraph_ptr_tptr + i);
  }
}

// Utility function that copies all results from an
// ngraph computation into the output NDArrays in mxnet
inline void result_to_NDArray(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& results,
    const std::vector<mxnet::OpReqType>& req,
    const std::vector<mxnet::NDArray>& outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (req[i] == mxnet::kNullOp) continue;

    const auto& element_type = getType(outputs[i].dtype());
    auto buffer_size = get_buffer_size(outputs[i].shape(), element_type.size());

    void* mxnet_ndarray = outputs[i].storage_handle().dptr;
    if (req[i] == mxnet::kAddTo) {
      void* ngraph_tv = malloc(buffer_size);
      results[i]->read(ngraph_tv, 0, buffer_size);

      if (element_type == ngraph::element::f32)
        result_plus_NDArray<float>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::f64)
        result_plus_NDArray<double>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::u8)
        result_plus_NDArray<uint8_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i8)
        result_plus_NDArray<int8_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i32)
        result_plus_NDArray<int32_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i64)
        result_plus_NDArray<int64_t>(mxnet_ndarray, ngraph_tv, buffer_size);

      free(ngraph_tv);
    } else {
      // TODO(adstraw): Add support for kWriteInplace
      results[i]->read(mxnet_ndarray, 0, buffer_size);
    }
  }
}
}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
