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

#ifndef MXNET_NGRAPH_NGRAPH_EMITTER_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_EMITTER_UTILS_H_

#include <string>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {

// slice data along a single axis statrting
// at a position and going a certain distance
// If the slice results in a single sample along the axis and flatten is true,
// the function will reshape to remove the unary axis.
NgraphNodePtr slice_data_on_axis(NgraphNodePtr data, size_t starting_loc,
                                 size_t step_size = 1, size_t axis = 0,
                                 bool flatten = true);

/**
 * Transforms input axis attribute with name in key based on MXNet convention (0
 * based index), where
 * negative values means indexing from the right.
 */
size_t transform_axis(int axis, int shape_size);

size_t get_default_transformed_axis(const NodePtr& node, const std::string& key,
                                    const int default_val,
                                    const int shape_size);

std::vector<size_t> get_default_transformed_axis(
    const NodePtr& node, const std::string& key,
    const ngraph::AxisVector& default_val, const int shape_size);

// cast result of op to given type
NgraphNodePtr cast_result(const NgraphNodePtr& op,
                          const ngraph::element::Type& type);
// clip utility function
NgraphNodePtr clip(const NgraphNodePtr& input, const float& min,
                   const float& max);
}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_EMITTER_UTILS_H_
