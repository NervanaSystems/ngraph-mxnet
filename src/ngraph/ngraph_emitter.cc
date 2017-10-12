#include "ngraph_emitter.h"
#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

// Compiter initialization
Emitter::Emitter() {
  // Create Operation Maps
  create_UnaryOps();
  create_BinaryOps();
  create_LayerOps();

  // Find all the valid operation names
  for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphLayerOps_) NgraphOps_.emplace_back(x.first);
}

// auto zero = std::make_shared<ngraph::op::ScalarConstant>(0);
// auto one = std::make_shared<ngraph::op::ScalarConstant>(1);
// unary op genrating function generator
void Emitter::create_UnaryOps() {
  // NgraphUnaryOps_["relu"] = [](const NgraphNodePtr& data){
  //   return std::make_shared<ngraph::op::Maximum>(data, zero);
  // };
  // NgraphUnaryOps_["sigmoid"] = [](const NgraphNodePtr& data){
  //   return (one / (one + std::make_shared<ngraph::op::Exp>(-data)));
  // };
  // NgraphUnaryOps_["softmax"] = [](const NgraphNodePtr& data){
  //   return std::make_shared<ngraph::op::Divide>(one, data);
  // };
  // NgraphUnaryOps_["log_softmax"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  NgraphUnaryOps_["_copy"] = [](const NgraphNodePtr& data){
    return data; //TODO: Return this as a reference. Does it actually need to be copied?
  };
  NgraphUnaryOps_["negative"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Negative>(data);
  };
  // NgraphUnaryOps_["reciprocal"] = [](const NgraphNodePtr& data){
  //   return one / data;
  // };
  NgraphUnaryOps_["abs"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Abs>(data);
  };
  // NgraphUnaryOps_["sign"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["round"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["rint"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  NgraphUnaryOps_["ceil"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Ceiling>(data);
  };
  NgraphUnaryOps_["floor"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Floor>(data);
  };
  // NgraphUnaryOps_["trunc"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["fix"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["square"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["rsqrt"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  NgraphUnaryOps_["exp"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Exp>(data);
  };
  NgraphUnaryOps_["log"] = [](const NgraphNodePtr& data) {
    return std::make_shared<ngraph::op::Log>(data);
  };
  // NgraphUnaryOps_["log10"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["log2"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["log1p"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["expm1"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["sin"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["cos"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["tan"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arcin"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arccos"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arctan"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["degrees"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["radians"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["sinh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["cosh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["tanh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arcsinh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arccosh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["arctanh"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["gamma"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
  // NgraphUnaryOps_["gammaln"] = [](const NgraphNodePtr& data){
  //   return ;
  // };
}

// binary op generating function generator
void Emitter::create_BinaryOps() {
  NgraphBinaryOps_["_plus"] = [](const NgraphNodePtr& lhs,
                                 const NgraphNodePtr& rhs) { 
    return (lhs + rhs);
  };
  NgraphBinaryOps_["_minus"] = [](const NgraphNodePtr& lhs,
                                  const NgraphNodePtr& rhs) {
    return (lhs - rhs);
  };
  NgraphBinaryOps_["_mul"] = [](const NgraphNodePtr& lhs,
                                const NgraphNodePtr& rhs) {
    return (lhs * rhs);
  };
  NgraphBinaryOps_["_div"] = [](const NgraphNodePtr& lhs,
                                const NgraphNodePtr& rhs) {
    return (lhs / rhs);
  };

  NgraphBinaryOps_["_power"] = [](const NgraphNodePtr& lhs,
                                  const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Power>(lhs, rhs);
  };
  NgraphBinaryOps_["_maximum"] = [](const NgraphNodePtr& lhs,
                                    const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Maximum>(lhs, rhs);
  };
  NgraphBinaryOps_["_minimum"] = [](const NgraphNodePtr& lhs,
                                    const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Minimum>(lhs, rhs);
  };
  // NgraphBinaryOps_["_hypot"] = [](const NgraphNodePtr& lhs,
  //                               const NgraphNodePtr& rhs,
  //                               ) {
  //   return ;
  // };
  NgraphBinaryOps_["_equal"] = [](const NgraphNodePtr& lhs,
                                  const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Equal>(lhs, rhs);
  };

  NgraphBinaryOps_["_not_equal"] = [](const NgraphNodePtr& lhs,
                                      const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::NotEqual>(lhs, rhs);
  };
  NgraphBinaryOps_["_greater"] = [](const NgraphNodePtr& lhs,
                                    const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Greater>(lhs, rhs);
  };
  NgraphBinaryOps_["_greater_equal"] = [](const NgraphNodePtr& lhs,
                                          const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::GreaterEq>(lhs, rhs);
  };
  NgraphBinaryOps_["_lesser"] = [](const NgraphNodePtr& lhs,
                                   const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::Less>(lhs, rhs);
  };
  NgraphBinaryOps_["_lesser_equal"] = [](const NgraphNodePtr& lhs,
                                         const NgraphNodePtr& rhs) {
    return std::make_shared<ngraph::op::LessEq>(lhs, rhs);
  };
}

// MXNet high level ops generating function
void Emitter::create_LayerOps() {
}
}  // end namespace ngraph
