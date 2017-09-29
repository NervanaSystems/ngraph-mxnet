// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#ifndef NGRAPH_PYCOMPILER_UTILS_H_
#define NGRAPH_PYCOMPILER_UTILS_H_

#include "ngraph_utils.h"
#include <sstream>
#include <iostream>

namespace ngraph {

using pyvec = std::vector<py::object>;

// utility to convert iterable object into a python tuple
// no error checking on the input, can fail miserably if user gives
// an incorrect input
template <typename T>
inline py::tuple createPyTuple(const T items) {
  py::tuple out = py::make_tuple();
  for (auto i : items) {
    out = out.attr("__add__")(py::make_tuple(i));
  }
  return out;
}

// get the number of axes in a ngraph op/placeholder
inline int num_axes(py::object data) {
  int i = 0;
  for (auto ax : data.attr("axes")) i += 1;
  return i;
}

// convoluted way to get the Nth axes of an ngraph placeholder/Op
// is there a better way to do this through the pybind API?
inline py::object getNthAxis(py::object data, int N) {
  int i = 0;
  for (auto ax : data.attr("axes")) {
    if (i < N) {
      i += 1;
      continue;
    }
    return ax.cast<py::object>();
  }
  throw ("N is larger than the number of axes in data");
}

// parse a list like (1, 2, 3) into a vector of ints [1,2,3]
inline std::vector<int> getInts(std::string input) {
  input = input.substr(1, input.size() - 2);
  std::stringstream ss(input);
  std::vector<int> vect;
  int i;
  while (ss >> i) {
    vect.push_back(i);

    if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
  }
  return vect;
}

}  // namespace ngraph
#endif  // UTILS_H_