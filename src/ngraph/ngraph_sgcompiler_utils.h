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

}  // namespace ngraph
#endif  // UTILS_H_