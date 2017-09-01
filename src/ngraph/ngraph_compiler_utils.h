#ifndef NGRAPH_COMPILER_UTILS_H_
#define NGRAPH_COMPILER_UTILS_H_

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
int num_axes(py::object data) {
  int i = 0;
  for (auto ax : data.attr("axes")) i += 1;
  return i;
}

// convoluted way to get the Nth axes of an ngraph placeholder/Op
// is there a better way to do this through the pybind API?
py::object getNthAxis(py::object data, int N) {
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

// function to remove modifiers frop op names
std::string clean_opname(std::string name) {
  for (std::string str : {"elemwise_", "broadcast_"})
    if (name.substr(0, str.size()) == str) name = name.substr(str.size());
  if (name == "_mul") name = "multiply";
  return name;
}


}  // namespace ngraph
#endif  // UTILS_H_