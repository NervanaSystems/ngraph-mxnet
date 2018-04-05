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

#include "ngraph_utils.h"

#include <algorithm>
#include <stdexcept>

#include <ngraph/serializer.hpp>
#include "nnvm/tuple.h"

using namespace std;

namespace ngraph_bridge {

std::ostream& operator<<(std::ostream& os, const ngraph::Shape& s) {
  return container_to_debug_stream(os, s);
}

std::ostream& operator<<(std::ostream& os, const nnvm::TShape& s) {
  return container_to_debug_stream(os, s);
}

std::ostream& operator<<(std::ostream& os, const ngraph::AxisSet& s) {
  return container_to_debug_stream(os, s, ", ", "{", "}");
}

ngraph::AxisSet shape_to_axis_set(const ngraph::Shape& s) {
  ngraph::AxisSet result;

  for (size_t i = 0; i < s.size(); ++i) {
    result.insert(i);
  }

  return result;
}

ngraph::AxisSet ngraph_remaining_axes(const NgraphNodePtr& n,
                                      const ngraph::AxisSet& a) {
  ngraph::AxisSet n_axes = shape_to_axis_set(n->get_shape());

  if (!std::includes(n_axes.begin(), n_axes.end(), a.begin(), a.end())) {
    std::ostringstream os;
    os << "NGRAPH_BRIDGE: In " << __PRETTY_FUNCTION__ << " : " << std::endl
       << "Trying to remove an axis that's not present in the node's shape:"
       << " shape=" << (n->get_shape()) << ", axis-set=" << a;
    throw std::runtime_error(os.str());
  }

  return ngraph::AxisSet(set_subtract(n_axes, a));
}

void dump_graph(std::shared_ptr<ngraph::Function> f, std::string src_loc,
                std::string filename_suffix) {
  std::stringstream fname;
  fname << "mxnet-ngraph";
  fname << "-" << f->get_name();

  if (!src_loc.empty()) {
    fname << "-" << src_loc;
  }

  if (!filename_suffix.empty()) {
    fname << "-" << filename_suffix;
  }

  fname << ".json";

  std::ofstream file;
  file.open(fname.str());
  file << ngraph::serialize(f) << std::endl;
  file.close();
}

bool has_vector_plus_axes_shape(const ngraph::Shape & s) {
  if (s.size() < 1) {
    return false;
  }

  bool already_found_big_axis = false;
  for (size_t axis_span : s) {
    if (axis_span == 0) {
      return false;
    }

    const bool is_big_axis = (axis_span > 1);

    if (is_big_axis) {
      if (already_found_big_axis) {
        return false;
      } else {
        already_found_big_axis = true;
      }
    }
  }

  return true;
}

ngraph::Shape get_vector_plus_axes_shape(
    const size_t rank,
    const size_t vector_axis,
    const size_t vector_length) {
  CHECK_GT(rank, 0);
  CHECK_GT(rank, vector_axis);
  CHECK_GT(vector_length, 0);

  ngraph::Shape s(rank, 1);
  s[vector_axis] = vector_length;

  return s;
}

NgraphNodePtr ensure_vector_only_shape(
    const NgraphNodePtr n) {
  CHECK(n);
  const ngraph::Shape & n_shape = n->get_shape();

  const size_t n_rank = n_shape.size();

  if (!has_vector_plus_axes_shape(n_shape)) {
    std::ostringstream os;
    os << "Tensor shape " << n_shape << " is not in vector-plus-axes form.";
    throw os.str();
  }

  if (n_rank == 1) {
    return n;
  } else {
    // We already know it's in vector-plus-axes form, so just count the number of elements.
    const size_t vector_length = ngraph::shape_size(n_shape);

    const ngraph::Shape output_shape{vector_length};
    const ngraph::AxisVector permute_order = pyrange(n_rank);

    return std::make_shared<ngraph::op::Reshape>(n, permute_order, output_shape);
  }
}

NgraphNodePtr ensure_vector_plus_axes_shape(
    const NgraphNodePtr n,
    const size_t n_vector_axis,
    const size_t output_rank,
    const size_t output_vector_axis) {
  CHECK(n);
  const ngraph::Shape & n_shape = n->get_shape();
  const size_t n_rank = n_shape.size();

  CHECK(n_vector_axis < n_rank);
  CHECK(n_rank <= output_rank);
  CHECK(output_vector_axis < output_rank);

  const size_t n_vector_length = n_shape[n_vector_axis];

  const ngraph::Shape output_shape = get_vector_plus_axes_shape(
      output_rank, output_vector_axis, n_vector_length);

  if (n_shape == output_shape) {
    return n;
  } else {
    const ngraph::AxisVector permute_order = pyrange(n_rank);
    const NgraphNodePtr ng_reshaped =
      std::make_shared<ngraph::op::Reshape>(n, permute_order, output_shape);
    return ng_reshaped;
  }
}


}  // namespace ngraph_bridge
