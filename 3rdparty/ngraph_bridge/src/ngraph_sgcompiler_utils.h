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

#ifndef MXNET_NGRAPH_NGRAPH_SGCOMPILER_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_SGCOMPILER_UTILS_H_

#include <mshadow/base.h>
#include <mshadow/tensor.h>

#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "ngraph_graph.h"

namespace ngraph_bridge {

inline const ngraph::element::Type& getType(int type) {
  static const std::map<int, const ngraph::element::Type*> typemap = {
      {mshadow::kFloat32, &ngraph::element::f32},
      {mshadow::kFloat64, &ngraph::element::f64},
      {mshadow::kUint8, &ngraph::element::u8},
      {mshadow::kInt8, &ngraph::element::i8},
      {mshadow::kInt32, &ngraph::element::i32},
      {mshadow::kInt64, &ngraph::element::i64}};

  auto ngraphType = typemap.find(type);
  if (ngraphType == typemap.end()) {
    throw std::runtime_error("NGRAPH_BRIDGE: type not supported");
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
inline nnvm::TShape convert_shapes(const ngraph::Shape& inshape) {
  return nnvm::TShape(inshape.begin(), inshape.end());
}

template <>
inline ngraph::Shape convert_shapes(const nnvm::TShape& inshape) {
  ngraph::Shape shape;
  for (const auto& s : inshape) {
    if (s < 0) {
      throw std::runtime_error(
          "NGRAPH_BRIDGE: After InferShape no shapes w/ negative dimensions");
    }
    shape.push_back(s);
  }
  return shape;
}

inline nnvm::TShape NShape_to_TShape(const ngraph::Shape& inshape) {
  return convert_shapes<ngraph::Shape, nnvm::TShape>(inshape);
}

// Only thing we're currently converting -> TShape to ngraph::Shape
inline ngraph::Shape TShape_to_NShape(const nnvm::TShape& inshape) {
  return convert_shapes<nnvm::TShape, ngraph::Shape>(inshape);
}

// Create a runtime typed constant defined by type, shape, and a string
// representing the number.
//
// Note that this function is NOT equivalent to nGraph's
// 'ngraph::op::Constant::create(...)' family of functions: This
// function uses the Broadacst op to achieve the desired shape.  This
// in some cases results in more efficient JIT compilation and runtime
// performance.
template <typename T>
inline std::shared_ptr<ngraph::Node> makeConstant(
    const ngraph::element::Type& type, const ngraph::Shape& shape,
    const T& num) {
  NgraphNodePtr val = std::make_shared<ngraph::op::Constant>(
      type, ngraph::Shape{}, std::vector<T>{num});

  if (shape.size() > 0) {
    ngraph::AxisSet axes;
    for (size_t i = 0; i < shape.size(); i++) axes.insert(i);
    val = std::make_shared<ngraph::op::Broadcast>(val, shape, axes);
  }

  return val;
}

// It's difficult to make a template specialization that handles string
// literals, so we'll keep things simple and use a non-template function
// overload instead.
inline std::shared_ptr<ngraph::Node> makeConstant(
    const ngraph::element::Type& type, const ngraph::Shape& shape,
    const char* const& num) {
  return makeConstant<std::string>(type, shape, std::string(num));
}

// Create a runtime typed constant from the type and shape of a node
// along with a string representing the number
inline std::shared_ptr<ngraph::Node> makeConstant(const NodePtr& node,
                                                  const std::string& num) {
  return makeConstant(getType(node->dtype_), TShape_to_NShape(node->shape_),
                      num);
}

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_SGCOMPILER_UTILS_H_
