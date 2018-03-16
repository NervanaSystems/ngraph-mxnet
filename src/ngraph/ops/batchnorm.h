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

#ifndef MXNET_NGRAPH_OPS_BATCHNORM_H_
#define MXNET_NGRAPH_OPS_BATCHNORM_H_

#include <tuple>

#include "ngraph_emitter.h"

namespace ngraph_bridge {

/// Return the shape one gets by starting with \param ng_in_data's shape,
/// and then changing to 1 the span of every dimension <i>other than</i>
/// the "channel" dimension.
///
/// For example, if the shape of \param ng_in_data is [N, C, H, W], this
/// function returns [1, C, 1, 1].
ngraph::Shape get_channel_only_keepdims_shape(const NgraphNodePtr& ng_in_data,
                                              const size_t channel_axis);

/// Create an nGraph subgraph that computes batch-norm...
/// - without using `ngraph::op::BatchNorm`, and
/// - using mean/variance supplied by caller.
///
/// \param ng_in_gamma_reshaped_or_null Iff not null, the normalization
/// formula will apply gamma correction.
///
/// \return A pointer to an nGraph node that computes the normalization of
/// \param ng_in_data.
NgraphNodePtr create_batchnorm_basic_computation_nodes(
    const NgraphNodePtr& ng_mean, const NgraphNodePtr& ng_variance,
    const NgraphNodePtr& ng_in_data, const NgraphNodePtr& ng_epsilon,
    const NgraphNodePtr& ng_in_gamma_reshaped_or_null,
    const NgraphNodePtr& ng_in_beta_reshaped);

/// Create an nGraph subgraph that computes batch-norm as well as batch-mean and
/// batch-variance.  Use those values of mean/variance for the batch-norm
/// operation.
/// The subgraph created by this function may or may not contain
/// `ngraph::op::BatchNorm`.
///
/// \return A tuple whose elements point to nGraph nodes that produce,
/// respectively:
/// - the normalized input data,
/// - the per-channel means of the input data, and
/// - the per-channel variances of the input data.
std::tuple< NgraphNodePtr, NgraphNodePtr, NgraphNodePtr >
create_batchnorm_fprop_and_batch_stats_nodes(
    const NgraphNodePtr& ng_in_data,
    const size_t channel_axis, const NgraphNodePtr& ng_epsilon,
    const NgraphNodePtr& ng_in_gamma_reshaped_or_null,
    const NgraphNodePtr& ng_in_beta_reshaped);

/// Create an nGraph subgraph that computes an updated moving-mean value.
NgraphNodePtr create_batchnorm_recalculate_moving_mean_nodes(
    const NgraphNodePtr& ng_ones,
    const NgraphNodePtr& ng_in_moving_mean_reshaped,
    const NgraphNodePtr& ng_batch_mean, const NgraphNodePtr& ng_momentum);

/// Create an nGraph subgraph that computes an updated moving-variance value.
NgraphNodePtr create_batchnorm_recalculate_moving_variance_nodes(
    const NgraphNodePtr& ng_ones,
    const NgraphNodePtr& ng_in_moving_variance_reshaped,
    const NgraphNodePtr& ng_batch_variance, const NgraphNodePtr& ng_momentum);

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_OPS_BATCHNORM_H_
