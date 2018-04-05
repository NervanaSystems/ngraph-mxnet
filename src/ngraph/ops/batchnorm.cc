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

#include <ngraph/op/get_output_element.hpp>

#include "ngraph_utils.h"
#include "ngraph_emitter.h"
#include "ngraph_sgcompiler_utils.h"

using std::make_shared;
using ngraph::builder::make_with_numpy_broadcast;

namespace ngraph_bridge {

static NgraphNodePtr create_batchnorm_basic_computation_nodes(
    const NgraphNodePtr& ng_mean, const NgraphNodePtr& ng_variance,
    const NgraphNodePtr& ng_in_data, const NgraphNodePtr& ng_epsilon,
    const NgraphNodePtr& ng_in_gamma_reshaped_or_null,
    const NgraphNodePtr& ng_in_beta_reshaped) {
  const NgraphNodePtr denom =
      make_shared<ngraph::op::Sqrt>(ng_variance + ng_epsilon);

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

std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>
create_batchnorm_training_without_ngraph_bn_op(
    const float epsilon,
    const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta,
    const NgraphNodePtr ng_in_data,
    const size_t channel_axis) {
  const ngraph::Shape & batch_data_shape = ng_in_data->get_shape();
  const size_t batch_data_rank = batch_data_shape.size();

  const ngraph::element::Type et = ng_in_data->get_element_type();
  CHECK(ng_beta->get_element_type() == et);

  CHECK(channel_axis < batch_data_rank);
  const size_t channel_axis_length = batch_data_shape[ channel_axis ];

  const ngraph::Shape channel_vector_plus_axes_shape = get_vector_plus_axes_shape(
      batch_data_rank, channel_axis, channel_axis_length);

  NgraphNodePtr ng_normalized_batch;

  const NgraphNodePtr ng_batch_means =
    Emitter::ReduceAxes(ng_in_data, {channel_axis}, true, true, ngraph::builder::mean);

  const NgraphNodePtr ng_batch_variances =
    Emitter::ReduceAxes(ng_in_data, {channel_axis}, true, true,
        [](const std::shared_ptr<ngraph::Node>& node,
          const ngraph::AxisSet& axes) {
        return ngraph::builder::variance(node, axes);
        });

  const NgraphNodePtr ng_epsilon_shaped = makeConstant(et, channel_vector_plus_axes_shape, epsilon);

  const NgraphNodePtr ng_beta_shaped =
    ensure_vector_plus_axes_shape(ng_beta, 0, batch_data_rank, channel_axis);

  const NgraphNodePtr ng_gamma_shaped_or_null = ng_maybe_gamma
    ? ensure_vector_plus_axes_shape(ng_maybe_gamma, 0, batch_data_rank, channel_axis)
    : NgraphNodePtr{};

  ng_normalized_batch = create_batchnorm_basic_computation_nodes(
      ng_batch_means, ng_batch_variances, ng_in_data, ng_epsilon_shaped,
      ng_gamma_shaped_or_null, ng_beta_shaped);

  const NgraphNodePtr ng_batch_means_vector_shaped =
    ensure_vector_only_shape(ng_batch_means);

  const NgraphNodePtr ng_batch_variances_vector_shaped =
    ensure_vector_only_shape(ng_batch_variances);

  return std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>{
    ng_normalized_batch,
    ng_batch_means_vector_shaped,
    ng_batch_variances_vector_shaped};
}

NgraphNodePtr create_batchnorm_inference_without_ngraph_bn_op(
    const float epsilon,
    const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta,
    const NgraphNodePtr ng_in_data,
    const NgraphNodePtr ng_moving_mean,
    const NgraphNodePtr ng_moving_var,
    const size_t channel_axis) {
  const ngraph::Shape & batch_data_shape = ng_in_data->get_shape();
  const size_t batch_data_rank = batch_data_shape.size();

  CHECK(channel_axis < batch_data_rank);
  const size_t channel_axis_length = batch_data_shape[ channel_axis ];

  const NgraphNodePtr ng_mean_shaped =
    ensure_vector_plus_axes_shape(ng_moving_mean, 0, batch_data_rank, channel_axis);

  const NgraphNodePtr ng_var_shaped =
    ensure_vector_plus_axes_shape(ng_moving_var, 0, batch_data_rank, channel_axis);

  const ngraph::Shape channel_vector_plus_axes_shape = get_vector_plus_axes_shape(
      batch_data_rank, channel_axis, channel_axis_length);

  const ngraph::element::Type et = ng_in_data->get_element_type();
  CHECK(ng_beta->get_element_type() == et);

  const NgraphNodePtr ng_epsilon_shaped = makeConstant(et, channel_vector_plus_axes_shape, epsilon);

  const NgraphNodePtr denom = std::make_shared<ngraph::op::Sqrt>(ng_var_shaped + ng_epsilon_shaped);

  const NgraphNodePtr numerator =
    make_with_numpy_broadcast<ngraph::op::Subtract>(ng_in_data, ng_mean_shaped);

  NgraphNodePtr result =
    make_with_numpy_broadcast<ngraph::op::Divide>(numerator, denom);

  if (ng_maybe_gamma) {
    const NgraphNodePtr ng_gamma_shaped =
      ensure_vector_plus_axes_shape(ng_maybe_gamma, 0, batch_data_rank, channel_axis);

    const NgraphNodePtr ng_scale_by_gamma =
      make_with_numpy_broadcast<ngraph::op::Multiply>(result, ng_gamma_shaped);

    result = ng_scale_by_gamma;
  }

  const NgraphNodePtr ng_beta_shaped =
    ensure_vector_plus_axes_shape(ng_beta, 0, batch_data_rank, channel_axis);

  result = make_with_numpy_broadcast<ngraph::op::Add>(result, ng_beta_shaped);

  return result;
}

}  // namespace ngraph_bridge
