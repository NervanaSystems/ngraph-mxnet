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

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <iostream>
#include <iterator>
#include <ngraph/ngraph.hpp>
#include <sstream>
#include <vector>
#include "ngraph_graph.h"
namespace ngraph_bridge {

inline const ngraph::element::Type& getType(int type) {
  static const std::map<int, const ngraph::element::Type*> typemap = {
      {mshadow::kFloat32, &ngraph::element::Float32::element_type()},
      {mshadow::kFloat64, &ngraph::element::Float64::element_type()},
      {mshadow::kUint8, &ngraph::element::UInt8::element_type()},
      {mshadow::kInt8, &ngraph::element::Int8::element_type()},
      {mshadow::kInt32, &ngraph::element::Int32::element_type()},
      {mshadow::kInt64, &ngraph::element::Int64::element_type()}};

  auto ngraphType = typemap.find(type);
  if (ngraphType == typemap.end()) {
    throw "NGRAPH_BRIDGE: type not supported";
  }
  return *ngraphType->second;
}

// Template for converting shape objects via a range based for loop
template <typename Ti, typename To>
inline To convert_shapes(const Ti& inshape) {
  To shape;
  for (const auto& s : inshape) shape.push_back(s);
  return shape;
}

template <>
inline ngraph::Shape convert_shapes(const nnvm::TShape& inshape) {
  ngraph::Shape shape;
  for (const auto& s : inshape) {
    if (s < 0) {
      throw "NGRAPH_BRIDGE: After InferShape no shapes w/ negative dimensions";
    }
    shape.push_back(s);
  }
  return shape;
}

// Only thing we're currently converting -> TShape to ngraph::Shape
inline ngraph::Shape TShape_to_NShape(const nnvm::TShape& inshape) {
  return convert_shapes<nnvm::TShape, ngraph::Shape>(inshape);
}

// Create a runtime typed constant from the type and shape of a node
// along with a string representing the number
inline std::shared_ptr<ngraph::Node> makeConstant(const NodePtr& node,
                                                  const std::string& num) {
  const auto& et = getType(node->dtype_);
  auto shape = TShape_to_NShape(node->shape_);
  return std::make_shared<ngraph::op::Constant>(et, shape, num);
}

// This function expects the input string to be of the form
// "(1,2,3)" with optional spaces between the numbers, i.e.
// "(1,2 , 3)". This is the standard format MXNet uses to represent things
// like stride/padding/reshape ordering
template <typename T>
inline std::vector<T> GetIntVectorFromString(std::string input) {
  for (char c : {' ', ')', '(', ']', '['})
    input.erase(std::remove(input.begin(), input.end(), c), input.end());
  std::stringstream ss(input);
  std::vector<T> vect;
  T i;
  while (ss >> i) {
    vect.push_back(i);
    if (ss.peek() == ',') ss.ignore();
  }
  return vect;
}

}  // namespace ngraph_bridge
#endif  // UTILS_H_
