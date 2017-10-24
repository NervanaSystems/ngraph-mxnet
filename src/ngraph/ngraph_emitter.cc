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

#include "ngraph_emitter.h"
#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

// Compiter initialization
Emitter::Emitter() {
  // Create Operation Maps
  create_UnaryOps();
  create_BinaryOps();
  create_LayerOps();
}

// unary op genrating function generator
void Emitter::create_UnaryOps() {
  NgraphOpFuncs_["relu"] = [this](const NodePtr& node) {
    auto zero = makeConstant(node, "0");
    return std::make_shared<ngraph::op::Maximum>(op_map[node->inputs[0]], zero);
  };
  NgraphOpFuncs_["sigmoid"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return (one / (one + std::make_shared<ngraph::op::Exp>(
                             -op_map[node->inputs[0]])));
  };
  // NgraphOpFuncs_["softmax"] = [this](const NodePtr& node) {
  //   auto numer = std::make_shared<ngraph::op::Exp>(op_map[node->inputs[0]]);
  //   auto denom = std::make_shared<ngraph::op::Sum>(numer, ngraph::AxisSet{1});
  //   return ;
  // };
  // NgraphOpFuncs_["log_softmax"] = [this](const NodePtr& node){
  //   return ;
  // };
  NgraphOpFuncs_["_copy"] = [this](const NodePtr& node) {
    return op_map[node->inputs[0]]; //TODO: Return this as a reference. Does it actually need to be copied?
  };
  NgraphOpFuncs_["negative"] = [this](const NodePtr& node) {
    return -op_map[node->inputs[0]];
  };
  NgraphOpFuncs_["reciprocal"] = [this](const NodePtr& node){
    auto one = makeConstant(node, "1");
    return one / op_map[node->inputs[0]];
  };
  NgraphOpFuncs_["abs"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Abs>(op_map[node->inputs[0]]);
  };
  // NgraphOpFuncs_["sign"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["round"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["rint"] = [this](const NodePtr& node){
  //   return ;
  // };
  NgraphOpFuncs_["ceil"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Ceiling>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["floor"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Floor>(op_map[node->inputs[0]]);
  };
  // NgraphOpFuncs_["trunc"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["fix"] = [this](const NodePtr& node){
  //   return ;
  // };
  NgraphOpFuncs_["square"] = [this](const NodePtr& node) {
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]], two);
  };
  NgraphOpFuncs_["sqrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]],
                                               one / two);
  };
  NgraphOpFuncs_["rsqrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return one / std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]],
                                                     one / two);
  };
  NgraphOpFuncs_["cbrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto three = makeConstant(node, "3");
    return std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]],
                                               one / three);
  };
  NgraphOpFuncs_["rcbrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto three = makeConstant(node, "3");
    return one / std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]],
                                                     one / three);
  };
  NgraphOpFuncs_["exp"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Exp>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["log"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Log>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["log10"] = [this](const NodePtr& node){
    auto ten = makeConstant(node, "10");
    return std::make_shared<ngraph::op::Log>(op_map[node->inputs[0]]) / 
           std::make_shared<ngraph::op::Log>(ten);
  };
  NgraphOpFuncs_["log2"] = [this](const NodePtr& node){
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Log>(op_map[node->inputs[0]]) / 
           std::make_shared<ngraph::op::Log>(two);
  };
  // NgraphOpFuncs_["log1p"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["expm1"] = [this](const NodePtr& node){
  //   return ;
  // };
  NgraphOpFuncs_["sin"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Sin>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["cos"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Cos>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["tan"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Tan>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["arcsin"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Asin>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["arccos"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Acos>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["arctan"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Atan>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["sinh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Sinh>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["cosh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Cosh>(op_map[node->inputs[0]]);
  };
  NgraphOpFuncs_["tanh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Tanh>(op_map[node->inputs[0]]);
  };
  // NgraphOpFuncs_["arcsinh"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["arccosh"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["arctanh"] = [this](const NodePtr& node){
  //   return ;
  // };
  NgraphOpFuncs_["degrees"] = [this](const NodePtr& node){
    auto pi = makeConstant(node, "3.14159265359");
    auto oneeighty = makeConstant(node, "180");
    return op_map[node->inputs[0]] * (oneeighty / pi);
  };
  NgraphOpFuncs_["radians"] = [this](const NodePtr& node){
    auto pi = makeConstant(node, "3.14159265359");
    auto oneeighty = makeConstant(node, "180");
    return op_map[node->inputs[0]] * (pi / oneeighty);
  };
  // NgraphOpFuncs_["gamma"] = [this](const NodePtr& node){
  //   return ;
  // };
  // NgraphOpFuncs_["gammaln"] = [this](const NodePtr& node){
  //   return ;
  // };
}

AutoBroadcast Emitter::createAutoBroadcast(const NodePtr& node) {
  auto lhsNode = op_map[node->inputs[0]];
  auto lhsShape = TShape_to_NShape(node->inputs[0]->shape);
  auto rhsNode = op_map[node->inputs[1]];
  auto rhsShape = TShape_to_NShape(node->inputs[1]->shape);
  return AutoBroadcast(lhsNode, lhsShape, rhsNode, rhsShape);
}

// binary op generating function generator
void Emitter::create_BinaryOps() {
  NgraphOpFuncs_["_plus"] = [this](const NodePtr& node) { 
    return (op_map[node->inputs[0]] + op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_minus"] = [this](const NodePtr& node) {
    return (op_map[node->inputs[0]] - op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_mul"] = [this](const NodePtr& node) {
    return (op_map[node->inputs[0]] * op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_div"] = [this](const NodePtr& node) {
    return (op_map[node->inputs[0]] / op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_mod"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Remainder>(op_map[node->inputs[0]],
                                                   op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_power"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]],
                                               op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_maximum"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Maximum>(op_map[node->inputs[0]],
                                                 op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_minimum"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Minimum>(op_map[node->inputs[0]],
                                                 op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_hypot"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(
        (std::make_shared<ngraph::op::Power>(op_map[node->inputs[0]], two) +
         std::make_shared<ngraph::op::Power>(op_map[node->inputs[1]], two)),
        one / two);
  };
  NgraphOpFuncs_["_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Equal>(op_map[node->inputs[0]],
                                               op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_not_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::NotEqual>(op_map[node->inputs[0]],
                                                  op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_greater"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Greater>(op_map[node->inputs[0]],
                                                 op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_greater_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::GreaterEq>(op_map[node->inputs[0]],
                                                   op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_lesser"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Less>(op_map[node->inputs[0]],
                                              op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["_lesser_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::LessEq>(op_map[node->inputs[0]],
                                                op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["dot"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Dot>(op_map[node->inputs[0]],
                                             op_map[node->inputs[1]]);
  };
  NgraphOpFuncs_["broadcast_add"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return ab.lhs() + ab.rhs();
  };
  NgraphOpFuncs_["broadcast_sub"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return ab.lhs() - ab.rhs();
  };
  NgraphOpFuncs_["broadcast_mul"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return ab.lhs() * ab.rhs();
  };
  NgraphOpFuncs_["broadcast_div"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return ab.lhs() / ab.rhs();
  };
  NgraphOpFuncs_["broadcast_mod"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Remainder>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_power"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Power>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_maximum"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Maximum>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_minimum"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Minimum>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_hypot"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(
        (std::make_shared<ngraph::op::Power>(ab.lhs(), two) +
         std::make_shared<ngraph::op::Power>(ab.rhs(), two)),
        one / two);
  };
  NgraphOpFuncs_["broadcast_equal"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Equal>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_not_equal"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::NotEqual>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_greater"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Greater>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_greater_equal"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::GreaterEq>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_lesser"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::Less>(ab.lhs(), ab.rhs());
  };
  NgraphOpFuncs_["broadcast_lesser_equal"] = [this](const NodePtr& node) {
    auto ab = createAutoBroadcast(node);
    return std::make_shared<ngraph::op::LessEq>(ab.lhs(), ab.rhs());
  };
}

// MXNet high level ops generating function
void Emitter::create_LayerOps() {

  NgraphOpFuncs_["split"] = [this](const NodePtr& node) {

    size_t axis = 1;
    int num_outputs = 1;
    int index = node->multioutput_index;
    bool squeeze_axis = false;

    for (auto& kv : node->orig_node->attrs.dict) {
      if (kv.first == "num_outputs") num_outputs = std::stoi(kv.second);
      if (kv.first == "axis") axis = std::stoi(kv.second);
      if (kv.first == "squeeze_axis") squeeze_axis = std::stoi(kv.second);
    }

    auto upper = TShape_to_NShape(node->inputs[0]->shape);
    std::vector<size_t> lower(upper.size(), 0);

    lower[axis] = index * upper[axis] / num_outputs;
    upper[axis] = (index + 1) * upper[axis] / num_outputs;

    std::shared_ptr<ngraph::Node> op = std::make_shared<ngraph::op::Slice>(
        op_map[node->inputs[0]], lower, upper);

    if (squeeze_axis && ((upper[axis] - lower[axis]) == 1)) {
      std::vector<size_t> reshape;
      for (size_t i = 0; i < upper.size(); ++i)
        if (i != axis) reshape.push_back(upper[i]);

      ngraph::AxisVector order(upper.size());
      std::iota(order.begin(), order.end(), 0);

      op = std::make_shared<ngraph::op::Reshape>(op, order, reshape);
    }

    return op;
  };

  NgraphOpFuncs_["concat"] = [this](const NodePtr& node) {
    size_t axis = 1;
    
    for (auto& kv : node->orig_node->attrs.dict) 
      if (kv.first == "axis") axis = std::stoi(kv.second);

    std::vector<NgraphNodePtr> args;
    for (auto i : node->inputs) args.push_back(op_map[i]);

    return std::make_shared<ngraph::op::Concat>(args, axis);
  };

  NgraphOpFuncs_["flatten"] = [this](const NodePtr& node) {
    auto in_shape = TShape_to_NShape(node->inputs[0]->shape);
    auto out_shape = ngraph::Shape({in_shape[0], 1});
    for (size_t i = 1; i < in_shape.size(); ++i) out_shape[1] *= in_shape[i];
    ngraph::AxisVector order(in_shape.size());
    std::iota(order.begin(), order.end(), 0);
    return std::make_shared<ngraph::op::Reshape>(op_map[node->inputs[0]], order,
                                                 out_shape);
  };

  NgraphOpFuncs_["transpose"] = [this](const NodePtr& node) {
    auto in_shape = TShape_to_NShape(node->inputs[0]->shape);
    if (in_shape.size() != 2) throw;
    auto out_shape = ngraph::Shape({in_shape[1], in_shape[0]});
    return std::make_shared<ngraph::op::Reshape>(
        op_map[node->inputs[0]], ngraph::AxisVector{1, 0}, out_shape);
  };

  NgraphOpFuncs_["expand_dims"] = [this](const NodePtr& node) {
    size_t axis = 1;
    for (auto& kv : node->orig_node->attrs.dict) 
      if (kv.first == "axis") axis = std::stoi(kv.second);
    
    auto in_shape = TShape_to_NShape(node->inputs[0]->shape);

    ngraph::AxisVector order(in_shape.size());
    std::iota(order.begin(), order.end(), 0);
    
    auto out_shape = in_shape;
    out_shape.insert(out_shape.begin() + axis, 1);

    return std::make_shared<ngraph::op::Reshape>(op_map[node->inputs[0]], order,
                                                 out_shape);
  };
  
}

}  // end namespace ngraph

