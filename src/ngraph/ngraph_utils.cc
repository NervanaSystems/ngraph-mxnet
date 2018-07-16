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

#include <stdexcept>

#include <algorithm>
#include <ngraph/serializer.hpp>

#include "ngraph_utils.h"
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

std::ostream& operator<<(std::ostream& os, const ngraph::AxisVector& s) {
  return container_to_debug_stream(os, s);
}

std::ostream& operator<<(std::ostream& os, const ngraph::Strides& s) {
  return container_to_debug_stream(os, s);
}

std::ostream& operator<<(std::ostream& os, const ngraph::CoordinateDiff& s) {
  return container_to_debug_stream(os, s);
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

bool has_vector_plus_axes_shape(const ngraph::Shape& s) {
  if (s.size() < 1) {
    return false;
  }

  // Are all axis spans positive?
  if (std::find(s.begin(), s.end(), 0) != s.end()) {
    return false;
  }

  // Is there at most one axis with a span > 1?
  const auto long_axis_pred = [](const size_t& span) { return span > 1; };

  if (std::count_if(s.begin(), s.end(), long_axis_pred) > 1) {
    return false;
  }

  return true;
}

// If the vector-axis has length greater than 1, return its index.  Otherwise
// return 0.
// If 's' is not in vector-plus-axes form, throw an exception.
static size_t get_vector_axis_index(const ngraph::Shape& s) {
  if (!has_vector_plus_axes_shape(s)) {
    std::ostringstream os;
    os << "Shape " << s << " not in vector-plus-axes form.";
    throw os.str();
  }

  const auto long_axis_pred = [](const size_t& span) { return span > 1; };

  const auto iter = std::find_if(s.begin(), s.end(), long_axis_pred);
  if (iter == s.end()) {
    return 0;
  } else {
    return std::distance(s.begin(), iter);
  }
}

ngraph::Shape get_vector_plus_axes_shape(const size_t rank,
                                         const size_t vector_axis,
                                         const size_t vector_length) {
  CHECK_GT(rank, 0);
  CHECK_GT(rank, vector_axis);
  CHECK_GT(vector_length, 0);

  ngraph::Shape s(rank, 1);
  s[vector_axis] = vector_length;

  return s;
}

NgraphNodePtr ensure_vector_only_shape(const NgraphNodePtr n) {
  CHECK(n);
  const ngraph::Shape& n_shape = n->get_shape();

  const size_t n_rank = n_shape.size();

  if (!has_vector_plus_axes_shape(n_shape)) {
    std::ostringstream os;
    os << "Tensor shape " << n_shape << " is not in vector-plus-axes form.";
    throw os.str();
  }

  if (n_rank == 1) {
    return n;
  } else {
    // We already know it's in vector-plus-axes form, so just count the number
    // of elements.
    const size_t vector_length = ngraph::shape_size(n_shape);

    const ngraph::Shape output_shape{vector_length};
    const ngraph::AxisVector permute_order = pyrange(n_rank);

    return std::make_shared<ngraph::op::Reshape>(n, permute_order,
                                                 output_shape);
  }
}

NgraphNodePtr ensure_vector_plus_axes_shape(const NgraphNodePtr n,
                                            const size_t output_rank,
                                            const size_t output_vector_axis) {
  CHECK(n);
  const ngraph::Shape& n_shape = n->get_shape();
  const size_t n_rank = n_shape.size();

  CHECK(n_rank <= output_rank);
  CHECK(output_vector_axis < output_rank);

  const size_t n_vector_axis = get_vector_axis_index(n_shape);
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
