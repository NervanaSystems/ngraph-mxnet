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

#include "ops/batchnorm.h"

#include "ngraph_utils.h"

using ngraph::builder::make_with_numpy_broadcast;

namespace ngraph_bridge {

ngraph::Shape get_channel_only_keepdims_shape(const NgraphNodePtr& ng_in_data,
                                              const size_t channel_axis) {
  ngraph::Shape s = ng_in_data->get_shape();

  CHECK(channel_axis < s.size());

  for (size_t i = 0; i < s.size(); ++i) {
    if (i != channel_axis) {
      s[i] = 1;
    }
  }

  return s;
}

NgraphNodePtr create_batchnorm_basic_computation_nodes(
    const NgraphNodePtr& ng_mean, const NgraphNodePtr& ng_variance,
    const NgraphNodePtr& ng_in_data, const NgraphNodePtr& ng_epsilon,
    const NgraphNodePtr& ng_in_gamma_reshaped_or_null,
    const NgraphNodePtr& ng_in_beta_reshaped) {
  const NgraphNodePtr denom =
      std::make_shared<ngraph::op::Sqrt>(ng_variance + ng_epsilon);

  const NgraphNodePtr numerator =
      make_with_numpy_broadcast<ngraph::op::Subtract>(ng_in_data, ng_mean);

  const NgraphNodePtr result_simply_normalized =
      make_with_numpy_broadcast<ngraph::op::Divide>(numerator, denom);

  NgraphNodePtr result_maybe_with_gamma;
  if (ng_in_gamma_reshaped_or_null) {
    result_maybe_with_gamma = make_with_numpy_broadcast<ngraph::op::Multiply>(
        result_simply_normalized, ng_in_gamma_reshaped_or_null);
  } else {
    result_maybe_with_gamma = result_simply_normalized;
  }

  return make_with_numpy_broadcast<ngraph::op::Add>(
      result_maybe_with_gamma, ng_in_beta_reshaped);
}

std::tuple< NgraphNodePtr, NgraphNodePtr, NgraphNodePtr >
create_batchnorm_fprop_and_batch_stats_nodes(
    const NgraphNodePtr& ng_in_data,
    const size_t channel_axis, const NgraphNodePtr& ng_epsilon,
    const NgraphNodePtr& ng_in_gamma_reshaped_or_null,
    const NgraphNodePtr& ng_in_beta_reshaped) {
  const size_t in_data_rank = ng_in_data->get_shape().size();
  CHECK_GT(in_data_rank, channel_axis);

  const size_t num_channels = ng_in_data->get_shape()[channel_axis];
  CHECK_GT(num_channels, 0);

  const bool use_ngraph_mkldnn_kernel =
      (in_data_rank == 4) && (channel_axis == 1) &&
      (ng_in_data->get_element_type() == ngraph::element::f32);

  if (false) {
    // FIXME: Placeholder for NGMX-334 fix, which will sometimes use the new
    // ngraph::op::BatchNorm operator.
  } else {
    NgraphNodePtr ng_batch_mean = Emitter::ReduceAxes(ng_in_data, {channel_axis}, true,
                                           true, ngraph::builder::mean);

    NgraphNodePtr ng_batch_variance =
        Emitter::ReduceAxes(ng_in_data, {channel_axis}, true, true,
                           [](const std::shared_ptr<ngraph::Node>& node,
                              const ngraph::AxisSet& axes) {
                             return ngraph::builder::variance(node, axes);
                           });

    NgraphNodePtr ng_out_data = create_batchnorm_basic_computation_nodes(
        ng_batch_mean, ng_batch_variance, ng_in_data, ng_epsilon,
        ng_in_gamma_reshaped_or_null, ng_in_beta_reshaped);

    return std::make_tuple( ng_out_data, ng_batch_mean, ng_batch_variance );
  }
}

NgraphNodePtr create_batchnorm_recalculate_moving_mean_nodes(
    const NgraphNodePtr& ng_ones,
    const NgraphNodePtr& ng_in_moving_mean_reshaped,
    const NgraphNodePtr& ng_batch_mean, const NgraphNodePtr& ng_momentum) {
  return ng_in_moving_mean_reshaped * ng_momentum +
    ng_batch_mean * (ng_ones - ng_momentum);
}

NgraphNodePtr create_batchnorm_recalculate_moving_variance_nodes(
    const NgraphNodePtr& ng_ones,
    const NgraphNodePtr& ng_in_moving_variance_reshaped,
    const NgraphNodePtr& ng_batch_variance, const NgraphNodePtr& ng_momentum) {
  return ng_in_moving_variance_reshaped * ng_momentum +
                           ng_batch_variance * (ng_ones - ng_momentum);
}

}  // namespace ngraph_bridge
