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

#include "ngraph_emitter_utils.h"
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

NgraphNodePtr slice_data_on_axis(NgraphNodePtr data, size_t starting_loc,
                                 size_t step_size, size_t axis, bool flatten) {
  // slice data on given axis
  ngraph::Coordinate lower(data->get_shape().size(), 0);
  ngraph::Coordinate upper = data->get_shape();

  lower[axis] = starting_loc;
  upper[axis] = starting_loc + step_size;

  NgraphNodePtr slice = std::make_shared<ngraph::op::Slice>(data, lower, upper);

  if (flatten && (step_size == 1)) {
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < slice->get_shape().size(); ++i) {
      if (i != axis) {
        out_shape.push_back(slice->get_shape()[i]);
      }
    }
    slice = std::make_shared<ngraph::op::Reshape>(
        slice, pyrange(data->get_shape().size()), out_shape);
  }

  return slice;
}

size_t transform_axis(int axis, int shape_size) {
  assert(abs(axis) <= shape_size);
  // convert negative axis index to postive (counting from right per mxnet
  // convention)
  return axis < 0 ? shape_size + axis : axis;
}

size_t get_default_transformed_axis(const NodePtr& node, const std::string& key,
                                    const int default_val,
                                    const int shape_size) {
  return transform_axis(get_default(node, key, default_val), shape_size);
}

std::vector<size_t> get_default_transformed_axis(
    const NodePtr& node, const std::string& key,
    const ngraph::AxisVector& default_val, const int shape_size) {
  std::vector<int> values;
  for (auto val : default_val) {
    values.push_back(val);
  }
  values = get_default(node, key, values);

  std::vector<size_t> axes;
  for (size_t i = 0; i < values.size(); ++i) {
    axes.push_back(transform_axis(values[i], shape_size));
  }
  return axes;
}

NgraphNodePtr cast_result(const NgraphNodePtr& op,
                          const ngraph::element::Type& type) {
  return std::make_shared<ngraph::op::Convert>(op, type);
}

NgraphNodePtr clip(const NgraphNodePtr& input, const float& min,
                   const float& max) {
  auto shape = input->get_shape();
  auto dtype = input->get_element_type();
  const NgraphNodePtr a_min = makeConstant(dtype, shape, min);
  const NgraphNodePtr a_max = makeConstant(dtype, shape, max);
  return std::make_shared<ngraph::op::Maximum>(
      std::make_shared<ngraph::op::Minimum>(input, a_max), a_min);
}

}  // namespace ngraph_bridge

#endif
