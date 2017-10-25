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
#include <sstream>
#include <iostream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_graph.h"
namespace ngraph_bridge {

inline const ngraph::element::Type& getType(int type_flag){
  switch (type_flag) {
    // case mshadow::kFloat16:
    //   return ngraph::element::Float16::element_type();
    //   break;  
    case mshadow::kFloat32: 
      return ngraph::element::Float32::element_type();
      break;
    // case mshadow::kFloat64:
    //   return ngraph::element::Float64::element_type();
    //   break;  
    case mshadow::kUint8:
      return ngraph::element::UInt8::element_type();
      break;
    case mshadow::kInt8:
      return ngraph::element::Int8::element_type();
      break;
    case mshadow::kInt32:
      return ngraph::element::Int32::element_type();
      break;
    case mshadow::kInt64:
      return ngraph::element::Int64::element_type();
      break;
    default:
      throw ("type not supported");
  } 
  return ngraph::element::Float32::element_type();
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

template <typename Ti, typename To>
inline To convert_shapes(const Ti& inshape){
  To shape;
  for (const auto& s : inshape) shape.push_back(s);
  return shape;
}

inline ngraph::Shape TShape_to_NShape(const nnvm::TShape& inshape){
  return convert_shapes<nnvm::TShape, ngraph::Shape>(inshape);
}

inline std::shared_ptr<ngraph::Node> makeConstant(const NodePtr& node,
                                                  std::string num) {
  const auto& et = getType(node->dtype);
  auto shape = TShape_to_NShape(node->shape);
  return std::make_shared<ngraph::op::Constant>(et, shape, num);
}

using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

inline NgraphNodePtr NgraphTranspose(NgraphNodePtr node,
                                     ngraph::Shape in_shape) {
  //TODO: Support multidimensional Transpose
  if (in_shape.size() != 2) throw;
  auto out_shape = ngraph::Shape({in_shape[1], in_shape[0]});
  return std::make_shared<ngraph::op::Reshape>(node, ngraph::AxisVector{1, 0},
                                               out_shape);
}

}  // namespace ngraph
#endif  // UTILS_H_