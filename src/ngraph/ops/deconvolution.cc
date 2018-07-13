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
#include <vector>

#include "ops/deconvolution.h"
#include "../../operator/nn/deconvolution-inl.h"

#include "ngraph_emitter.h"
#include "ngraph_emitter_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

NgraphNodePtr create_deconvolution(const NgraphNodePtr& data,
                                   const NgraphNodePtr& filter,
                                   const ngraph::Shape& out_shape,
                                   const nnvm::NodePtr& orig_node) {
  const auto& param =
      nnvm::get<mxnet::op::DeconvolutionParam>(orig_node->attrs.parsed);
  const auto data_shape = data->get_shape();
  const auto filter_shape = filter->get_shape();

  auto n = data_shape.size() - 2;
  ngraph::CoordinateDiff pad(param.pad.begin(), param.pad.end());
  ngraph::Strides stride(param.stride.begin(), param.stride.end());
  ngraph::Strides dilate(param.dilate.begin(), param.dilate.end());
  auto num_group = param.num_group;
  if (pad.size() == 0) {
    pad = ngraph::CoordinateDiff(n, 0);
  }
  if (stride.size() == 0) {
    stride = ngraph::Strides(n, 1);
  }
  if (dilate.size() == 0) {
    dilate = ngraph::Strides(n, 1);
  }

  NgraphNodePtr conv;
  if (num_group == 1) {
    conv = std::make_shared<ngraph::op::ConvolutionBackpropData>(
        out_shape, filter, data, stride, dilate, pad, pad,
        ngraph::Strides(n, 1));
  } else {
    std::vector<NgraphNodePtr> convolutions(num_group);
    auto sliced_out_shape = out_shape;
    sliced_out_shape[1] /= num_group;
    for (size_t g = 0; g < num_group; ++g) {
      // slice data on channel_in
      size_t data_slice_step = data_shape[1] / num_group;
      size_t filter_slice_step = filter_shape[0] / num_group;
      auto data_slice = slice_data_on_axis(data, g * data_slice_step,
                                           data_slice_step, 1, false);
      auto filter_slice = slice_data_on_axis(filter, g * filter_slice_step,
                                             filter_slice_step, 0, false);
      // convolve sliced data and filter
      // N, channel_out/groups, d'1,...,d'n
      convolutions[g] = std::make_shared<ngraph::op::ConvolutionBackpropData>(
          sliced_out_shape, filter_slice, data_slice, stride, dilate, pad, pad,
          ngraph::Strides(n, 1));
    }

    // concatenate convolutions on channel_out
    // N, channel_out, d'1,...,d'n
    conv = std::make_shared<ngraph::op::Concat>(convolutions, 1);
  }

  return conv;
}

}  // namespace ngraph_bridge
