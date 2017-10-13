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
  // NgraphOpFuncs_["softmax"] = [this](const NodePtr& node){
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
  // NgraphOpFuncs_["_hypot"] = [this](const NodePtr& lhs) {
  //   return ;
  // };
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
}

// MXNet high level ops generating function
void Emitter::create_LayerOps() {
}
}  // end namespace ngraph
