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
    const mxnet::TBlob& input, bool copy = false) {
  auto shape = TShape_to_NShape(input.shape_);
  const auto& element_type = getType(input.type_flag_);
  auto TV = element_type.make_primary_tensor_view(shape);

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
inline ValueVector make_ngraph_placeholders(
    const std::vector<mxnet::TBlob>& inputs, bool copy_data = false) {
  ValueVector out;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(out),
                 [copy_data](const mxnet::TBlob& tb) {
                   return TBlob_to_TensorView(tb, copy_data);
                 });
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

  auto TV = std::dynamic_pointer_cast<TensorView>(result);
  TV->read(p, 0, buffer_size);
}

}  // namespace ngraph_bridge
