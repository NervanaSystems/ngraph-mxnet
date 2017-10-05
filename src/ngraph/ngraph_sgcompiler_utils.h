#ifndef NGRAPH_PYCOMPILER_UTILS_H_
#define NGRAPH_PYCOMPILER_UTILS_H_

#include <sstream>
#include <iostream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_graph.h"
namespace ngraph_bridge {

inline const ngraph::element::Type& getType(int type_flag){
  if (type_flag == 0){
    return ngraph::element::Float32::element_type();
  } else if (type_flag == 1) {
    throw "Float64 not supported by ngraph";
    // return ngraph::element::Float64::element_type();
  } else if (type_flag == 2) {
    throw "Float16 not supported by ngraph";
    // return ngraph::element::Float16::element_type();
  } else if (type_flag == 3) {
    return ngraph::element::UInt8::element_type();
  } else if (type_flag == 4) {
    return ngraph::element::Int32::element_type();
  } else if (type_flag == 5) {
    return ngraph::element::Int8::element_type();
  } else if (type_flag == 6) {
    return ngraph::element::Int64::element_type();
  }
  throw "data type not known";
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

inline ngraph::Shape TShape_to_NShape(const nnvm::TShape& inshape){
  ngraph::Shape shape;
  for (const auto& s : inshape) 
    shape.push_back(s);
  return shape;
}

}  // namespace ngraph
#endif  // UTILS_H_