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
#include <cassert>
#include <stdexcept>

using namespace std;

namespace ngraph_bridge {

std::ostream& operator<<(std::ostream& os, const ngraph::Shape& s) {
  os << "[";
  for (size_t d : s) {
    os << " " << d;
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ngraph::AxisSet& s) {
  os << "{";
  for (size_t a : s) {
    os << " " << a;
  }
  os << "}";
  return os;
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
  assert(n);

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

}  // namespace ngraph_bridge
