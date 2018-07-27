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
#include <string>
#include <vector>

#include "ops/pooling.h"
#include "../../operator/nn/pool.h"
#include "../../operator/nn/pooling-inl.h"

#include "ngraph_emitter_utils.h"
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {
using namespace mxnet::op;

struct PoolingParams {
  PoolingParams(const NodePtr& node, const NgraphNodePtr& input) {
    const auto& param =
        nnvm::get<mxnet::op::PoolingParam>(node->orig_node_->attrs.parsed);
    if (param.pooling_convention == pool_enum::kFull) {
      pooling_convention = "full";
    } else {
      pooling_convention = "valid";
    }
    global_pool = param.global_pool;

    auto input_shape = input->get_shape();
    // first two tensor axes are batch and channel, rest are image channels
    // get the number of image channels for pooling
    auto pool_dim = input_shape.size() - 2;
    auto default_ones = std::vector<size_t>(pool_dim, 1);
    auto default_zeros = std::vector<size_t>(pool_dim, 0);

    kernel = std::vector<size_t>(param.kernel.begin(), param.kernel.end());
    stride = std::vector<size_t>(param.stride.begin(), param.stride.end());
    pad = std::vector<size_t>(param.pad.begin(), param.pad.end());
    if (kernel.size() == 0) {
      kernel = default_ones;
    }
    if (stride.size() == 0) {
      stride = default_ones;
    }
    if (pad.size() == 0) {
      pad = default_zeros;
    }

    // if global pooling is true, reset the pooling kernel to the
    // input image size
    if (global_pool) {
      kernel = std::vector<size_t>(input_shape.begin() + 2, input_shape.end());
    }
  }

  std::string pooling_convention;
  bool global_pool;
  std::vector<size_t> kernel;
  std::vector<size_t> stride;
  std::vector<size_t> pad;
};

std::vector<size_t> asymetric_padding(const ngraph::Shape& input_shape,
                                      const PoolingParams& params) {
  auto top_pad = params.pad;
  if (params.pooling_convention == "full") {
    for (size_t i = 2; i < input_shape.size(); ++i) {
      size_t padded_dim = input_shape[i] + 2 * top_pad[i - 2];
      size_t stride = params.stride[i - 2];
      // calculate extra padding
      auto num_strides = static_cast<size_t>(
          ceil(static_cast<float>(padded_dim - params.kernel[i - 2]) /
               static_cast<float>(stride)));
      size_t extra_pad =
          num_strides * stride + params.kernel[i - 2] - padded_dim;
      top_pad[i - 2] += extra_pad;
    }
  }
  return top_pad;
}

NgraphNodePtr max_pooling(const NodePtr& node, const NgraphNodePtr& input) {
  auto params = PoolingParams(node, input);

  return std::make_shared<ngraph::op::MaxPool>(
      input, params.kernel, params.stride, params.pad,
      asymetric_padding(input->get_shape(), params));
}

NgraphNodePtr avg_pooling(const NodePtr& node, const NgraphNodePtr& input) {
  auto params = PoolingParams(node, input);

  return std::make_shared<ngraph::op::AvgPool>(
      input, params.kernel, params.stride, params.pad,
      asymetric_padding(input->get_shape(), params), true);
}

NgraphNodePtr sum_pooling(const NodePtr& node, const NgraphNodePtr& input) {
  auto params = PoolingParams(node, input);

  // Compute the sum-pool by first computing the avg-pool, and then
  // element-wise multiply (the resulting vector by each element of the
  // resulting tensor) with (the number of elements in the pooling window).
  // We do this because nGraph++ doesn't directly support sum-pooling.

  const size_t num_window_elements = ngraph::shape_size(params.kernel);

  const auto avg_pool_op = std::make_shared<ngraph::op::AvgPool>(
      input, params.kernel, params.stride, params.pad,
      asymetric_padding(input->get_shape(), params), true);

  const auto coeff_op = ngraph_bridge::makeConstant(
      avg_pool_op->get_element_type(), avg_pool_op->get_shape(),
      std::to_string(num_window_elements));

  auto mul_op = std::make_shared<ngraph::op::Multiply>(avg_pool_op, coeff_op);

  return mul_op;
}

NgraphNodePtr create_pooling(const NodePtr& node, const NgraphNodePtr& input) {
  NgraphNodePtr op;
  const auto& param =
      nnvm::get<mxnet::op::PoolingParam>(node->orig_node_->attrs.parsed);
  auto type = param.pool_type;
  if (type == pool_enum::kMaxPooling) {
    op = max_pooling(node, input);
  } else if (type == pool_enum::kAvgPooling) {
    op = avg_pooling(node, input);
  } else if (type == pool_enum::kSumPooling) {
    op = sum_pooling(node, input);
  } else {
    throw std::runtime_error("NGRAPH_BRIDGE: Unsupported Pooling Type");
  }
  return op;
}

}  // namespace ngraph_bridge
