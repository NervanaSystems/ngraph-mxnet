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

// Create an subgraph of nGraph nodes that's functionally equivalent to nGraph's
// training-version
// BatchNorm op, but which doesn't actually use that particular nGraph op.
// This function exists as a crutch until nGraph's BatchNorm op is fully
// implemented.
//
// All of the nGraph nodes created by this function support autodiff.
std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>
create_batchnorm_training_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    size_t channel_axis);

// Create an subgraph of nGraph nodes that's functionally equivalent to nGraph's
// inference-version
// BatchNorm op, but which doesn't actually use that particular nGraph op.
// This function exists as a crutch until nGraph's BatchNorm op is fully
// implemented.
//
// All of the nGraph nodes created by this function support autodiff.
NgraphNodePtr create_batchnorm_inference_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const NgraphNodePtr ng_moving_mean, const NgraphNodePtr ng_moving_var,
    size_t channel_axis);

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_OPS_BATCHNORM_H_
