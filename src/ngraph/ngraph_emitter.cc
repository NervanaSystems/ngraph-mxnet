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

#include "ngraph_emitter.h"
#include "ngraph_emitter_utils.h"
#include "ngraph_utils.h"

#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ngraph/op/get_output_element.hpp>
#include <ngraph/op/reverse_sequence.hpp>
#include "ngraph_sgcompiler_utils.h"
#include "ops/batchnorm.h"
#include "ops/deconvolution.h"
#include "ops/pooling.h"
#include "ops/slice.h"

namespace ngraph_bridge {

// Compiter initialization
Emitter::Emitter() { setExeMode(GraphExeMode::kInfer); }

void Emitter::setExeMode(GraphExeMode exe_mode) {
  exe_mode_ = exe_mode;
  InitOpFuncs();
}

void Emitter::InitOpFuncs() {
  ngraph_op_funcs_.clear();
  aux_op_map_.clear();
  // clear the op_map_ and placeholder_order
  ClearOpMap();
  // Create Operation Maps
  CreateUnaryOps();
  CreateBinaryOps();
  CreateLayerOps();
  CreateLossOps();
  UnsupportedOps();
}

void Emitter::ClearOpMap() {
  // delete the temporary storage
  op_map_.clear();
  placeholder_order_.clear();
}

/**
 * Performs a reduction on the node using the input axes.
 * @param axes list of axis to operate on
 * @param exclude if true should use node axes not listed in axes parameter,
 *  otherwise use input axes for the reduction operation.
 * @param keepdims if true the result will be reshaped to have the same
 *  dimension as the node (where reduction axes will have size 1), otherwise
 *  leave the resulting shape produced by func unchanged.
 * @param func reduction operation
 * @return resulting node of the reduction operation
 */
NgraphNodePtr Emitter::ReduceAxes(
    const NgraphNodePtr& node, ngraph::AxisVector axes, bool exclude,
    bool keepdims,
    const std::function<NgraphNodePtr(const NgraphNodePtr&,
                                      const ngraph::AxisSet&)>& func) {
  ngraph::AxisSet reduction_axes;
  if (axes.size() == 0) {
    for (size_t i = 0; i < node->get_shape().size(); ++i)
      reduction_axes.insert(i);
  } else if (exclude) {
    for (size_t i = 0; i < node->get_shape().size(); ++i)
      if (!in_vec(axes, i)) reduction_axes.insert(i);
  } else {
    for (auto i : axes) reduction_axes.insert(i);
  }
  auto output = func(node, reduction_axes);
  if (output->get_shape().size() == 0) {
    output = std::make_shared<ngraph::op::Reshape>(output, ngraph::AxisVector{},
                                                   ngraph::Shape{1});
  }

  if (keepdims) {
    auto reshape = node->get_shape();
    for (auto i : reduction_axes) reshape[i] = 1;

    output = std::make_shared<ngraph::op::Reshape>(
        output, pyrange(output->get_shape().size()), reshape);
  }

  if (output->get_shape() == ngraph::Shape()) {
    output = std::make_shared<ngraph::op::Reshape>(output, ngraph::AxisVector{},
                                                   ngraph::Shape{1});
  }
  return output;
}

NgraphNodePtr Emitter::ReduceAxes(
    const NodePtr& node,
    const std::function<NgraphNodePtr(const NgraphNodePtr&,
                                      const ngraph::AxisSet&)>& func) const {
  auto input = op_map_.at(node->inputs_[0]);
  auto axes = pyrange(input->get_shape().size());
  return ReduceAxes(
      input, get_default_transformed_axis(node, "axis", axes, axes.size()),
      get_default(node, "exclude", false), get_default(node, "keepdims", false),
      func);
}

// unary op function generator
void Emitter::CreateUnaryOps() {
  ngraph_op_funcs_["Activation"] = [this](const NodePtr node) {
    auto act_type = node->orig_node_->attrs.dict["act_type"];
    return ngraph_op_funcs_[act_type](node);
  };
  ngraph_op_funcs_["LeakyReLU"] = [this](const NodePtr& node) {
    const std::string act_type =
        get_default(node, "act_type", std::string("leaky"));
    const float slope = get_default(node, std::string("slope"), 0.25f);
    NgraphNodePtr ng_result;

    if (act_type == "leaky") {
      // f(x) = slope * x for x < 0
      // f(x) = x for x >= 0

      // The documentation for MXnet's LeakyReLU op doesn't state that slop must
      // be positive.
      // But that is the convention, and assuming it allows a simple,
      // efficicient implementation.
      // If we need to relax this assumption, our implementation must change.
      if (slope < 0) {
        std::ostringstream os;
        os << "NGRAPH_BRIDGE: LeakyReLU: 'slope' is assumed to be "
              "non-negative, but its value"
           << " is " << slope;
        throw std::runtime_error(os.str());
      }

      NgraphNodePtr ng_slope = makeConstant(node, std::to_string(slope));
      NgraphNodePtr ng_input0 = op_map_[node->inputs_[0]];
      ng_result = std::make_shared<ngraph::op::Maximum>(ng_input0 * ng_slope,
                                                        ng_input0);
    } else {
      // ngraph_bridge::Compiler::CheckInNgraph() has a check that should
      // prevent this code from
      // ever executing, but we'll want this check in place even after we remove
      // the test in
      // CheckInNgraph().  --cconvey
      std::ostringstream os;
      os << "NGRAPH_BRIDGE: LeakyReLU: No support yet for act_type '"
         << act_type << "'";
      throw std::runtime_error(os.str());
    }

    return ng_result;
  };
  ngraph_op_funcs_["relu"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Relu>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["softrelu"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return std::make_shared<ngraph::op::Log>(
        one + std::make_shared<ngraph::op::Exp>(op_map_[node->inputs_[0]]));
  };
  ngraph_op_funcs_["sigmoid"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return (one / (one + std::make_shared<ngraph::op::Exp>(
                             -op_map_[node->inputs_[0]])));
  };
  ngraph_op_funcs_["softmax"] = [this](const NodePtr& node) {
    auto axis =
        get_default_transformed_axis(node, "axis", 1, node->shape_.ndim());
    return std::make_shared<ngraph::op::Softmax>(op_map_[node->inputs_[0]],
                                                 ngraph::AxisSet{axis});
  };
  // };
  // ngraph_op_funcs_["log_softmax"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["SoftmaxActivation"] = [this](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto in_shape = input->get_shape();

    auto mode = get_default(node, "mode", std::string("instance"));

    ngraph::AxisSet axes;
    if (mode == "channel") {
      axes = ngraph::AxisSet{1};
    } else {
      axes = ngraph::AxisSet{in_shape.size() - 1};
    }

    return std::make_shared<ngraph::op::Softmax>(input, axes);
  };
  ngraph_op_funcs_["_copy"] = [this](const NodePtr& node) {
    return op_map_[node->inputs_[0]];
  };
  ngraph_op_funcs_["negative"] = [this](const NodePtr& node) {
    return -op_map_[node->inputs_[0]];
  };
  ngraph_op_funcs_["reciprocal"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return one / op_map_[node->inputs_[0]];
  };
  ngraph_op_funcs_["abs"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Abs>(op_map_[node->inputs_[0]]);
  };
  // ngraph_op_funcs_["sign"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["round"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["rint"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["ceil"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Ceiling>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["floor"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Floor>(op_map_[node->inputs_[0]]);
  };
  // ngraph_op_funcs_["trunc"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["fix"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["square"] = [this](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    return input * input;
  };
  ngraph_op_funcs_["sqrt"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Sqrt>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["rsqrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return one / std::make_shared<ngraph::op::Sqrt>(op_map_[node->inputs_[0]]);
  };
  // TODO(mbrookhart): MXNet's tests assume that this returns a matrix of nans
  // if some of the inputs
  // are negative. No idea why, it should be a mix of valid and nan data, which
  // is what ngraph returns
  /*
  ngraph_op_funcs_["cbrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto three = makeConstant(node, "3");
    return std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]],
                                               one / three);
  };
  ngraph_op_funcs_["rcbrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto three = makeConstant(node, "3");
    return one / std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]],
                                                     one / three);
  };
  */
  ngraph_op_funcs_["exp"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Exp>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["log"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Log>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["log10"] = [this](const NodePtr& node) {
    auto ten = makeConstant(node, "10");
    return std::make_shared<ngraph::op::Log>(op_map_[node->inputs_[0]]) /
           std::make_shared<ngraph::op::Log>(ten);
  };
  ngraph_op_funcs_["log2"] = [this](const NodePtr& node) {
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Log>(op_map_[node->inputs_[0]]) /
           std::make_shared<ngraph::op::Log>(two);
  };
  // ngraph_op_funcs_["log1p"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["expm1"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["sin"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Sin>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["cos"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Cos>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["tan"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Tan>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["arcsin"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Asin>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["arccos"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Acos>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["arctan"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Atan>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["sinh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Sinh>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["cosh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Cosh>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["tanh"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Tanh>(op_map_[node->inputs_[0]]);
  };
  // TODO(mbrookhart): Arc trig autodiff not implemented
  /*
  ngraph_op_funcs_["arcsinh"] = [this](const NodePtr& node){
    return ;
  };
  ngraph_op_funcs_["arccosh"] = [this](const NodePtr& node){
    return ;
  };
  ngraph_op_funcs_["arctanh"] = [this](const NodePtr& node){
    return ;
  };
  */
  ngraph_op_funcs_["_zeros"] = [this](const NodePtr& node) {
    return makeConstant(node, "0");
  };
  ngraph_op_funcs_["zeros_like"] = [this](const NodePtr& node) {
    return makeConstant(node->inputs_[0], "0");
  };
  ngraph_op_funcs_["degrees"] = [this](const NodePtr& node) {
    auto pi = makeConstant(node, "3.14159265359");
    auto oneeighty = makeConstant(node, "180");
    return op_map_[node->inputs_[0]] * (oneeighty / pi);
  };
  ngraph_op_funcs_["radians"] = [this](const NodePtr& node) {
    auto pi = makeConstant(node, "3.14159265359");
    auto oneeighty = makeConstant(node, "180");
    return op_map_[node->inputs_[0]] * (pi / oneeighty);
  };
  ngraph_op_funcs_["reverse"] = [this](const NodePtr& node) -> NgraphNodePtr {
    auto axes = get_default(node, "axis", std::vector<size_t>());
    ngraph::AxisSet axis_set;
    for (auto x : axes) {
      axis_set.insert(x);
    }
    return std::make_shared<ngraph::op::Reverse>(op_map_[node->inputs_[0]],
                                                 axis_set);
  };
  ngraph_op_funcs_["reshape"] = [this](const NodePtr& node) -> NgraphNodePtr {
    auto new_shape = TShape_to_NShape(node->shape_);

    auto input = op_map_[node->inputs_[0]];
    // ngraph++'s reshape wouldn't like an empty shape
    if (new_shape.size() == 0) {
      return std::make_shared<ngraph::op::Constant>(input->get_element_type(),
                                                    ngraph::Shape{}, "0");
    }

    return std::make_shared<ngraph::op::Reshape>(
        input, pyrange(input->get_shape().size()), new_shape);
  };
  ngraph_op_funcs_["swapaxes"] = [this](const NodePtr& node) -> NgraphNodePtr {
    auto input = op_map_[node->inputs_[0]];

    size_t dim1 = get_default(node, "dim1", 0);
    size_t dim2 = get_default(node, "dim2", 0);

    auto axes = pyrange(input->get_shape().size());
    std::swap(axes[dim1], axes[dim2]);

    auto new_shape = TShape_to_NShape(node->shape_);

    return std::make_shared<ngraph::op::Reshape>(input, axes, new_shape);
  };

  // ngraph_op_funcs_["gamma"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["gammaln"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["cast"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Convert>(op_map_[node->inputs_[0]],
                                                 getType(node->dtype_));
  };
  ngraph_op_funcs_["stop_gradient"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::StopGradient>(
        op_map_[node->inputs_[0]]);
  };

  //----------------------------- Reduce Ops ----------------------------//
  ngraph_op_funcs_["norm"] = [this](const NodePtr& node) {
    auto norm_ord1 = [](const NgraphNodePtr& node,
                        const ngraph::AxisSet& reduction_axes) {
      return std::make_shared<ngraph::op::Sum>(
          std::make_shared<ngraph::op::Abs>(node), reduction_axes);
    };
    auto ord = get_default(node, "ord", 2);
    if (ord == 1) {
      return ReduceAxes(node, norm_ord1);
    } else {
      return ReduceAxes(node, ngraph::builder::l2_norm);
    }
  };
  ngraph_op_funcs_["mean"] = [this](const NodePtr& node) {
    return ReduceAxes(node, ngraph::builder::mean);
  };
  ngraph_op_funcs_["sum"] = [this](const NodePtr& node) {
    auto create_sum = [](const NgraphNodePtr& node,
                         const ngraph::AxisSet& reduction_axes) {
      return std::make_shared<ngraph::op::Sum>(node, reduction_axes);
    };
    return ReduceAxes(node, create_sum);
  };
}

// autobroadcast factory function to avoid code copy
template <class op>
std::shared_ptr<ngraph::Node> Emitter::CreateAutoBroadcast(
    const NodePtr& node) {
  auto arg0 = op_map_[node->inputs_[0]];
  auto arg1 = op_map_[node->inputs_[1]];
  return ngraph::builder::make_with_numpy_broadcast<op>(arg0, arg1);
}
template <class op>
std::shared_ptr<ngraph::Node> Emitter::CreateScalarOp(const NodePtr& node) {
  auto arg0 = op_map_[node->inputs_[0]];
  auto arg1 = makeConstant(node, get_default(node, "scalar", std::string("0")));
  return std::make_shared<op>(arg0, arg1);
}

// binary op generating function generator
void Emitter::CreateBinaryOps() {
  ngraph_op_funcs_["_plus"] = [this](const NodePtr& node) {
    return (op_map_[node->inputs_[0]] + op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_minus"] = [this](const NodePtr& node) {
    return (op_map_[node->inputs_[0]] - op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_mul"] = [this](const NodePtr& node) {
    return (op_map_[node->inputs_[0]] * op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_div"] = [this](const NodePtr& node) {
    return (op_map_[node->inputs_[0]] / op_map_[node->inputs_[1]]);
  };
  // TODO(mbrookhart): Remainder not implemented
  // ngraph_op_funcs_["_mod"] = [this](const NodePtr& node) {
  //   return std::make_shared<ngraph::op::Remainder>(op_map_[node->inputs_[0]],
  //                                                  op_map_[node->inputs_[1]]);
  // };
  ngraph_op_funcs_["_power"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]],
                                               op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_maximum"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Maximum>(op_map_[node->inputs_[0]],
                                                 op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_minimum"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Minimum>(op_map_[node->inputs_[0]],
                                                 op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_hypot"] = [this](const NodePtr& node) {
    auto A = op_map_[node->inputs_[0]];
    auto B = op_map_[node->inputs_[1]];
    return std::make_shared<ngraph::op::Sqrt>((A * A) + (B * B));
  };
  // TODO(aemani): remove cast_result if ngraph supports same type
  ngraph_op_funcs_["_equal"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::Equal>(op_map_[node->inputs_[0]],
                                            op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  ngraph_op_funcs_["_not_equal"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::NotEqual>(op_map_[node->inputs_[0]],
                                               op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  ngraph_op_funcs_["_greater"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::Greater>(op_map_[node->inputs_[0]],
                                              op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  ngraph_op_funcs_["_greater_equal"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::GreaterEq>(op_map_[node->inputs_[0]],
                                                op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  ngraph_op_funcs_["_lesser"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::Less>(op_map_[node->inputs_[0]],
                                           op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  ngraph_op_funcs_["_lesser_equal"] = [this](const NodePtr& node) {
    return cast_result(
        std::make_shared<ngraph::op::LessEq>(op_map_[node->inputs_[0]],
                                             op_map_[node->inputs_[1]]),
        getType(node->dtype_));
  };
  auto dot_transpose = [this](const NodePtr& node, NgraphNodePtr left,
                              NgraphNodePtr right) {
    if (get_default(node, "transpose_a", false)) {
      auto N = left->get_shape().size();
      auto order = pyrange(1, N);
      order.push_back(0);
      left = ngraph::builder::numpy_transpose(left, order);
    }

    if (get_default(node, "transpose_b", false)) {
      auto N = right->get_shape().size();
      auto order = pyrange(N - 1);
      order.insert(order.begin(), N - 1);
      right = ngraph::builder::numpy_transpose(right, order);
    }
    return std::pair<NgraphNodePtr, NgraphNodePtr>{left, right};
  };
  ngraph_op_funcs_["dot"] = [this, dot_transpose](const NodePtr& node) {
    NgraphNodePtr left = op_map_[node->inputs_[0]];
    NgraphNodePtr right = op_map_[node->inputs_[1]];
    auto args = dot_transpose(node, left, right);

    const NgraphNodePtr dot =
        std::make_shared<ngraph::op::Dot>(args.first, args.second, 1);
    const size_t dot_rank = dot->get_shape().size();

    // A scalar value in nGraph has shape {}, but in MXnet it has shape {1}...
    NgraphNodePtr dot_shaped;
    if (dot_rank == 0) {
      ngraph::AxisVector input_order{};
      ngraph::Shape output_shape{1};
      dot_shaped =
          std::make_shared<ngraph::op::Reshape>(dot, input_order, output_shape);
    } else {
      dot_shaped = dot;
    }

    return dot_shaped;
  };
  ngraph_op_funcs_["batch_dot"] = [this, dot_transpose](const NodePtr& node) {
    NgraphNodePtr left = op_map_[node->inputs_[0]];
    NgraphNodePtr right = op_map_[node->inputs_[1]];

    auto left_shape = left->get_shape();
    auto right_shape = right->get_shape();

    size_t groups = left->get_shape()[0];
    std::vector<NgraphNodePtr> dots(groups);

    for (size_t g = 0; g < groups; ++g) {
      auto sliced_left = slice_data_on_axis(left, g);
      auto sliced_right = slice_data_on_axis(right, g);

      auto args = dot_transpose(node, sliced_left, sliced_right);
      auto dot = std::make_shared<ngraph::op::Dot>(args.first, args.second, 1);

      std::vector<size_t> out_shape{1};
      out_shape.insert(out_shape.end(), dot->get_shape().begin(),
                       dot->get_shape().end());

      dots[g] = std::make_shared<ngraph::op::Reshape>(
          dot, pyrange(sliced_left->get_shape().size()), out_shape);
    }

    // concatenate dots on batch channel
    return std::make_shared<ngraph::op::Concat>(dots, 0);
  };
  ngraph_op_funcs_["reshape_like"] = [this](const NodePtr& node) {
    auto arg0 = op_map_[node->inputs_[0]];
    auto reshape = op_map_[node->inputs_[1]]->get_shape();
    return std::make_shared<ngraph::op::Reshape>(
        arg0, pyrange(arg0->get_shape().size()), reshape);
  };
  ngraph_op_funcs_["_plus_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Add>(node);
  };
  ngraph_op_funcs_["_minus_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Subtract>(node);
  };
  ngraph_op_funcs_["_rminus_scalar"] = [this](const NodePtr& node) {
    auto arg0 =
        makeConstant(node, get_default(node, "scalar", std::string("0")));
    auto arg1 = op_map_[node->inputs_[0]];
    return std::make_shared<ngraph::op::Subtract>(arg0, arg1);
  };
  ngraph_op_funcs_["_mul_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Multiply>(node);
  };
  ngraph_op_funcs_["_div_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Divide>(node);
  };
  ngraph_op_funcs_["_rdiv_scalar"] = [this](const NodePtr& node) {
    auto arg0 =
        makeConstant(node, get_default(node, "scalar", std::string("0")));
    auto arg1 = op_map_[node->inputs_[0]];
    return std::make_shared<ngraph::op::Divide>(arg0, arg1);
  };
  ngraph_op_funcs_["_equal_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::Equal>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["_not_equal_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::NotEqual>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["_greater_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::Greater>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["_greater_equal_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::GreaterEq>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["_lesser_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::Less>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["_lesser_equal_scalar"] = [this](const NodePtr& node) {
    return cast_result(CreateScalarOp<ngraph::op::LessEq>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_add"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Add>(node);
  };
  ngraph_op_funcs_["broadcast_sub"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Subtract>(node);
  };
  ngraph_op_funcs_["broadcast_mul"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Multiply>(node);
  };
  ngraph_op_funcs_["broadcast_div"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Divide>(node);
  };
  // TODO(mbrookhart): Remainder not implemented in CPU
  // ngraph_op_funcs_["broadcast_mod"] = [this](const NodePtr& node) {
  //   return CreateAutoBroadcast<ngraph::op::Remainder>(node);
  // };
  ngraph_op_funcs_["broadcast_power"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Power>(node);
  };
  ngraph_op_funcs_["broadcast_maximum"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Maximum>(node);
  };
  ngraph_op_funcs_["broadcast_minimum"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Minimum>(node);
  };
  ngraph_op_funcs_["broadcast_hypot"] = [this](const NodePtr& node) {
    auto A = op_map_[node->inputs_[0]];
    auto B = op_map_[node->inputs_[1]];
    return std::make_shared<ngraph::op::Sqrt>(
        ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>((A * A),
                                                                    (B * B)));
  };
  // TODO(aemani): remove cast_result if ngraph enables same type result
  ngraph_op_funcs_["broadcast_equal"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::Equal>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_not_equal"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::NotEqual>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_greater"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::Greater>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_greater_equal"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::GreaterEq>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_lesser"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::Less>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_lesser_equal"] = [this](const NodePtr& node) {
    return cast_result(CreateAutoBroadcast<ngraph::op::LessEq>(node),
                       getType(node->dtype_));
  };
  ngraph_op_funcs_["broadcast_to"] =
      [this](const NodePtr& node) -> NgraphNodePtr {
    auto input = op_map_[node->inputs_[0]];
    const auto& input_shape = input->get_shape();
    ngraph::Shape output_shape{
        get_default(node, "shape", std::vector<size_t>{})};
    ngraph::AxisSet broadcast_axes;
    ngraph::Shape proxy_shape;
    assert(ngraph::shape_size(input_shape) == ngraph::shape_size(output_shape));
    // ngraph::op::broadcast does not allow in-place broadcast (must add a
    // new axis), so we reshape the input and eliminate axes with length 1,
    // then add these axes back with proper output length through
    // ngraph::op::broadcast.
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (output_shape[i] == 0) {
        output_shape[i] = input_shape[i];
      }
      if (input_shape[i] != output_shape[i]) {
        // only axis with dim 1 can be broadcasted, this should already been
        // checked by mxnet front end, but check in case it's being called
        // by other ops.
        assert(input_shape[i] == 1);
        broadcast_axes.insert(i);
      } else {
        proxy_shape.push_back(input_shape[i]);
      }
    }
    NgraphNodePtr input_reshape = std::make_shared<ngraph::op::Reshape>(
        input, pyrange(input_shape.size()), proxy_shape);
    return std::make_shared<ngraph::op::Broadcast>(input_reshape, output_shape,
                                                   broadcast_axes);
  };
  ngraph_op_funcs_["smooth_l1"] = [this](const NodePtr& node) {
    /* Smooth L1 Loss is a loss specific for R-CNN franchise training
     * Smooth L1 Loss function:
     * f(x) = 0.5 * (sigma * x) ^ 2,     |x| < 1 / sigma^2
     *      = |x| - 0.5 / (sigma ^ 2), otherwise
     * When sigma = 1, it is equivalent to the Huber loss, evaluated at
     * delta = 1.
     * smooth_l1_loss = w_out * f(w_in * x)
     * with w_in, w_out provided by input_data.
     */
    auto input = op_map_[node->inputs_[0]];
    auto sigma =
        makeConstant(node, get_default(node, "scalar", std::string("0")));
    auto sigma_sq = sigma * sigma;
    auto inv_sigma_sq = std::make_shared<ngraph::op::Divide>(
        makeConstant(node, "1.0"), sigma_sq);

    // check if input is greater than inv_sigma_sq
    auto is_input_gt_inv_sigma_sq =
        std::make_shared<ngraph::op::Greater>(input, inv_sigma_sq);

    auto half = makeConstant(node, "0.5");
    auto half_inv_sigma_sq = half * inv_sigma_sq;

    // 0.5 * (sigma * x) ^ 2
    auto result_input_sq = half * input * input * sigma_sq;

    // cant use abs as we need -input depending on input < -inv_sigma_sq
    // x - 0.5 / (sigma ^ 2)
    auto result_input_gt = input - half_inv_sigma_sq;
    // -x - 0.5 / (sigma ^ 2)
    auto result_input_lt = -input - half_inv_sigma_sq;

    // select result according to formula
    auto is_input_lt = std::make_shared<ngraph::op::Less>(input, -inv_sigma_sq);
    auto result_input = std::make_shared<ngraph::op::Select>(
        is_input_lt, result_input_lt, result_input_sq);

    return std::make_shared<ngraph::op::Select>(is_input_gt_inv_sigma_sq,
                                                result_input_gt, result_input);
  };
  ngraph_op_funcs_["SequenceMask"] = [this](const NodePtr& node) {
    auto data = op_map_[node->inputs_[0]];

    // if sequence lengths specified
    auto use_sequence_length = get_default(node, "use_sequence_length", false);
    if (use_sequence_length) {
      auto sequence_lengths = op_map_[node->inputs_[1]];

      // default: sequence axis = 0; batch axis = 1
      // alternative:  sequence axis = 1; batch axis = 0
      auto sequence_axis = get_default(node, "axis", 0);
      auto batch_axis = (sequence_axis == 0) ? 1 : 0;

      // create a mask from sequence lengths, same shape as node
      auto mask = ngraph::builder::tensor_mask<ngraph::op::Less>(
          sequence_lengths, sequence_axis, batch_axis, data->get_shape(), 0);

      // create value constant, same shape as node
      auto value = get_default(node, "value", std::string("0"));
      auto value_constant =
          makeConstant(ngraph::element::f32, data->get_shape(), value);

      // data[mask==False] = value
      data = std::make_shared<ngraph::op::Select>(mask, data, value_constant);
    }
    return data;
  };
  ngraph_op_funcs_["SequenceLast"] = [this](const NodePtr& node) {
    auto data = op_map_[node->inputs_[0]];

    size_t sequence_axis = get_default(node, "axis", 0);

    // if sequence lengths specified
    auto use_sequence_length = get_default(node, "use_sequence_length", false);
    if (use_sequence_length) {
      auto sequence_lengths = op_map_[node->inputs_[1]];

      // default: sequence axis = 0; batch axis = 1
      // alternative:  sequence axis = 1; batch axis = 0
      size_t batch_axis = (sequence_axis == 0) ? 1 : 0;

      // create a mask from sequence lengths, same shape as node
      auto mask = ngraph::builder::tensor_mask<ngraph::op::Equal>(
          sequence_lengths, sequence_axis, batch_axis, data->get_shape(), 1);

      // convert the mask to 0/1 from True/False
      auto convert_mask =
          std::make_shared<ngraph::op::Convert>(mask, data->get_element_type());

      // zero out non-last locations
      data = data * convert_mask;

      // collapse to only last locations
      data = std::make_shared<ngraph::op::Sum>(data,
                                               ngraph::AxisSet{sequence_axis});
    } else {
      data = slice_data_on_axis(data, data->get_shape().at(sequence_axis) - 1,
                                1, sequence_axis, true);
    }
    return data;
  };
}

// MXNet high level ops generating function
void Emitter::CreateLayerOps() {
  // In mxnet, split takes a tensor and creates multiple tensors from
  // equal slices along 1 axis. The compiler creates a subgraph where
  // each of those outputs is a single node.  This function creates
  // the slice op for making each tensor.
  ngraph_op_funcs_["split"] = [this](const NodePtr& node) {
    size_t axis = get_default_transformed_axis(node, "axis", 1,
                                               node->inputs_[0]->shape_.ndim());
    int num_outputs = get_default(node, "num_outputs", 1);
    int index = node->multi_output_index_;
    bool squeeze_axis = get_default(node, "squeeze_axis", false);

    auto input = op_map_[node->inputs_[0]];
    auto input_shape = input->get_shape();
    size_t slice_step = input_shape[axis] / num_outputs;
    return slice_data_on_axis(input, index * slice_step, slice_step, axis,
                              squeeze_axis && (slice_step == 1));
  };

  // slice op
  ngraph_op_funcs_["slice"] = [this](const NodePtr& node) -> NgraphNodePtr {
    NgraphNodePtr ng_slice =
        create_slice_op(op_map_[node->inputs_[0]], node->orig_node_->attrs);
    return ng_slice;
  };

  // stack takes a list of tensors of equal shape and
  // concatenates them along a given axis expanded for each input
  ngraph_op_funcs_["stack"] = [this](const NodePtr& node) {
    // get the concat axis
    size_t axis = get_default_transformed_axis(
        node, "axis", 0, node->inputs_[0]->shape_.ndim() + 1);
    auto shape = op_map_[node->inputs_[0]]->get_shape();
    shape.insert(shape.begin() + axis, 1);
    // grab input ngraph nodes and Reshape them
    std::vector<NgraphNodePtr> args;
    for (auto i : node->inputs_) {
      args.push_back(std::make_shared<ngraph::op::Reshape>(
          op_map_[i], pyrange(shape.size() - 1), shape));
    }

    // run concat
    return std::make_shared<ngraph::op::Concat>(args, axis);
  };

  // concat takes a list of tensors of equal shape and
  // concatenates them along a given axis
  ngraph_op_funcs_["concat"] = [this](const NodePtr& node) {
    // get the concat axis
    size_t axis = get_default_transformed_axis(node, "dim", 1,
                                               node->inputs_[0]->shape_.ndim());
    // grab in input ngraph nodes
    std::vector<NgraphNodePtr> args;
    for (auto i : node->inputs_) args.push_back(op_map_[i]);

    // run concat
    return std::make_shared<ngraph::op::Concat>(args, axis);
  };

  // tile takes a tensor and replicates it along
  // a given set of axes a given number of times
  ngraph_op_funcs_["tile"] = [this](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto shape = input->get_shape();
    // get the concat axis
    std::vector<size_t> reps;
    reps = get_default(node, "reps", reps);
    // promote the shape if it's smaller
    while (shape.size() < reps.size()) {
      shape.insert(shape.begin(), 1);
    }
    // propote the reps if it's smaller
    while (reps.size() < shape.size()) {
      reps.insert(reps.begin(), 1);
    }
    // reshape the input if needed
    if (shape != input->get_shape()) {
      input = std::make_shared<ngraph::op::Reshape>(
          input, pyrange(input->get_shape().size()), shape);
    }
    // tile along all the axes
    for (size_t i = 0; i < reps.size(); ++i) {
      std::vector<NgraphNodePtr> args;
      for (size_t j = 0; j < reps[i]; ++j) args.push_back(input);
      input = std::make_shared<ngraph::op::Concat>(args, i);
    }

    return input;
  };
  // where is mxnet's version of select
  ngraph_op_funcs_["where"] = [this](const NodePtr& node) {
    auto condition = op_map_[node->inputs_[0]];
    auto x = op_map_[node->inputs_[1]];
    auto y = op_map_[node->inputs_[2]];
    if (condition->get_shape() != x->get_shape()) {
      ngraph::AxisSet axes;
      for (size_t i = 1; i < x->get_shape().size(); ++i) {
        axes.insert(i);
      }
      condition = std::make_shared<ngraph::op::Broadcast>(condition,
                                                          x->get_shape(), axes);
    }
    condition = std::make_shared<ngraph::op::Convert>(condition,
                                                      ngraph::element::boolean);
    return std::make_shared<ngraph::op::Select>(condition, x, y);
  };
  // Fully connected is the main linear transformation layer in MXNet
  // it implements dot(data, W.T) + b
  ngraph_op_funcs_["FullyConnected"] = [this](const NodePtr& node) {
    auto X = op_map_[node->inputs_[0]];
    auto W = op_map_[node->inputs_[1]];

    auto flatten = get_default(node, "flatten", true);
    auto no_bias = get_default(node, "no_bias", false);

    if (flatten && X->get_shape().size() != 2) {
      ngraph::Shape flat_shape{X->get_shape()[0], 1};
      for (size_t i = 1; i < X->get_shape().size(); ++i) {
        flat_shape[1] *= X->get_shape()[i];
      }
      X = std::make_shared<ngraph::op::Reshape>(
          X, pyrange(X->get_shape().size()), flat_shape);
    } else if (X->get_shape().back() != W->get_shape()[1]) {
      ngraph::Shape shape = X->get_shape();
      shape.push_back(W->get_shape()[1]);
      X = std::make_shared<ngraph::op::Reshape>(
          X, pyrange(X->get_shape().size()), shape);
    }

    NgraphNodePtr dot = std::make_shared<ngraph::op::Dot>(
        X, ngraph::builder::numpy_transpose(W));

    if (!no_bias) {
      auto beta = op_map_[node->inputs_[2]];
      auto shape = beta->get_shape();
      if (flatten && shape.size() > 1) {
        beta = std::make_shared<ngraph::op::Reshape>(
            beta, pyrange(shape.size()),
            ngraph::Shape{std::accumulate(shape.begin(), shape.end(), 1ul,
                                          std::multiplies<size_t>())});
      }
      dot = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(dot,
                                                                        beta);
    }
    return dot;
  };

  // clip op
  ngraph_op_funcs_["clip"] = [this](const NodePtr& node) {
    return clip(op_map_[node->inputs_[0]], get_default(node, "a_min", 0.0f),
                get_default(node, "a_max", 0.0f));
  };

  // sgd_update op
  ngraph_op_funcs_["sgd_update"] = [this](const NodePtr& node) {
    auto weight = op_map_[node->inputs_[0]];
    auto grad = op_map_[node->inputs_[1]];
    auto shape = weight->get_shape();
    auto dtype = weight->get_element_type();

    const float clip_gradient = get_default(node, "clip_gradient", -1.0f);
    const NgraphNodePtr ng_rescale_grad =
        makeConstant(dtype, shape, get_default(node, "rescale_grad", 1.0f));
    const NgraphNodePtr ng_wd =
        makeConstant(dtype, shape, get_default(node, "wd", 0.0f));
    const NgraphNodePtr ng_lr =
        makeConstant(dtype, shape, get_default(node, "lr", 0.0f));
    const NgraphNodePtr one = makeConstant(dtype, shape, 1.0f);

    NgraphNodePtr scale_grad;

#if MXNET_USE_NGRAPH_DISTRIBUTED
    grad = std::make_shared<ngraph::op::AllReduce>(grad);
#endif
    if (clip_gradient >= 0.0f) {
      scale_grad = clip(ng_rescale_grad * grad, -clip_gradient, clip_gradient);
    } else {
      scale_grad = ng_rescale_grad * grad;
    }

    return (one - ng_lr * ng_wd) * weight - (ng_lr * scale_grad);
  };

  // sgd_mom_update op
  ngraph_op_funcs_["sgd_mom_update"] = [this](const NodePtr& node) {
    auto weight = op_map_[node->inputs_[0]];
    auto grad = op_map_[node->inputs_[1]];
    auto mom = op_map_[node->inputs_[2]];
    auto shape = weight->get_shape();
    auto dtype = weight->get_element_type();

    const float clip_gradient = get_default(node, "clip_gradient", -1.0f);
    const NgraphNodePtr ng_rescale_grad =
        makeConstant(dtype, shape, get_default(node, "rescale_grad", 1.0f));
    const NgraphNodePtr ng_wd =
        makeConstant(dtype, shape, get_default(node, "wd", 0.0f));
    const NgraphNodePtr ng_lr =
        makeConstant(dtype, shape, get_default(node, "lr", 0.0f));
    const NgraphNodePtr ng_mom =
        makeConstant(dtype, shape, get_default(node, "momentum", 0.0f));
    const NgraphNodePtr one = makeConstant(dtype, shape, 1.0f);

    NgraphNodePtr scale_grad;

#if MXNET_USE_NGRAPH_DISTRIBUTED
    grad = std::make_shared<ngraph::op::AllReduce>(grad);
#endif
    if (clip_gradient >= 0.0f) {
      scale_grad = clip(ng_rescale_grad * grad, -clip_gradient, clip_gradient);
    } else {
      scale_grad = ng_rescale_grad * grad;
    }

    auto mom_update =
        (ng_mom * mom) - (ng_lr * ng_wd * weight) - (ng_lr * scale_grad);
    aux_op_map_[node->inputs_[2]] = mom_update;
    return weight + mom_update;
  };

  // flatten converts an array of shape (x0, x1, x2, ...)
  // to an array of shape (x0, x1*x2*...)
  ngraph_op_funcs_["flatten"] = [this](const NodePtr& node) {
    auto in_shape = TShape_to_NShape(node->inputs_[0]->shape_);
    auto out_shape = ngraph::Shape({in_shape[0], 1});
    out_shape[1] = std::accumulate(in_shape.begin() + 1, in_shape.end(), 1,
                                   std::multiplies<int>());

    return std::make_shared<ngraph::op::Reshape>(
        op_map_[node->inputs_[0]], pyrange(in_shape.size()), out_shape);
  };

  // Implement transpose with a utility function that returns
  // a reshape op. Not ideal, we should have a ngraph transpose op
  ngraph_op_funcs_["transpose"] = [this](const NodePtr& node) {
    auto axes_order = get_default(node, "axes", ngraph::AxisVector());
    return ngraph::builder::numpy_transpose(op_map_[node->inputs_[0]],
                                            axes_order);
  };

  // expand dims inserts an axis of length 1 somewhere in the tensor shape
  ngraph_op_funcs_["expand_dims"] = [this](const NodePtr& node) {
    size_t axis = get_default(node, "axis", 1);

    auto in_shape = TShape_to_NShape(node->inputs_[0]->shape_);

    // copy the shape and insert a 1 at the axis position to expand the
    // dimension
    auto out_shape = in_shape;
    out_shape.insert(out_shape.begin() + axis, 1);

    return std::make_shared<ngraph::op::Reshape>(
        op_map_[node->inputs_[0]], pyrange(in_shape.size()), out_shape);
  };

  // batch norm operation
  ngraph_op_funcs_["BatchNorm"] = [this](const NodePtr& node) -> NgraphNodePtr {
    enum InputName { kData = 0, kGamma, kBeta, kMovingMean, kMovingVar };
    NgraphNodePtr ng_in_data = op_map_[node->inputs_[kData]];
    NgraphNodePtr ng_in_gamma = op_map_[node->inputs_[kGamma]];
    NgraphNodePtr ng_in_beta = op_map_[node->inputs_[kBeta]];
    NgraphNodePtr ng_in_moving_mean = op_map_[node->inputs_[kMovingMean]];
    NgraphNodePtr ng_in_moving_var = op_map_[node->inputs_[kMovingVar]];
    const int data_shape_size =
        static_cast<int>(ng_in_data->get_shape().size());

    // Default Batch norm parameters
    const float eps = get_default(node, "eps", 0.001f);
    const float momentum = get_default(node, "momentum", 0.9f);
    const bool fix_gamma = get_default(node, "fix_gamma", true);
    const bool use_global_stats = get_default(node, "use_global_stats", false);
    // zero based channel axis
    const size_t channel_axis =
        get_default_transformed_axis(node, "axis", 1, node->shape_.ndim());

    const NgraphNodePtr ng_maybe_gamma =
        fix_gamma ? NgraphNodePtr{} : ng_in_gamma;

    const NgraphNodePtr ng_actual_gamma =
        fix_gamma ? makeConstant(ng_in_moving_mean->get_element_type(),
                                 ng_in_moving_mean->get_shape(), 1)
                  : ng_in_gamma;

    using ngraph::builder::make_with_numpy_broadcast;

    const bool ngraph_bn_op_available = (data_shape_size == 4) &&
                                        (channel_axis == 1) &&
                                        (node->dtype_ == mshadow::kFloat32);

    //----------------------------------------------------------------------------------------------
    // Traditional training mode...
    //----------------------------------------------------------------------------------------------
    if ((exe_mode_ == GraphExeMode::kTrain) && (!use_global_stats)) {
      NgraphNodePtr ng_normalized_data;
      NgraphNodePtr ng_batch_mean;
      NgraphNodePtr ng_batch_var;

      if (ngraph_bn_op_available) {
        const NgraphNodePtr BN = std::make_shared<ngraph::op::BatchNorm>(
            eps, ng_actual_gamma, ng_in_beta, ng_in_data);
        ng_normalized_data =
            std::make_shared<ngraph::op::GetOutputElement>(BN, 0);
        ng_batch_mean = std::make_shared<ngraph::op::GetOutputElement>(BN, 1);
        ng_batch_var = std::make_shared<ngraph::op::GetOutputElement>(BN, 2);
      } else {
        std::tie(ng_normalized_data, ng_batch_mean, ng_batch_var) =
            create_batchnorm_training_without_ngraph_bn_op(
                eps, ng_maybe_gamma, ng_in_beta, ng_in_data, channel_axis);
      }

      const NgraphNodePtr ng_one =
          makeConstant(ng_in_moving_mean->get_element_type(),
                       ng_in_moving_mean->get_shape(), 1);

      const NgraphNodePtr ng_momentum =
          makeConstant(ng_in_moving_var->get_element_type(),
                       ng_in_moving_var->get_shape(), momentum);

      aux_op_map_[node->inputs_[kMovingMean]] =
          ng_in_moving_mean * ng_momentum +
          ng_batch_mean * (ng_one - ng_momentum);

      aux_op_map_[node->inputs_[kMovingVar]] =
          ng_in_moving_var * ng_momentum +
          ng_batch_var * (ng_one - ng_momentum);

      return ng_normalized_data;
    }

    //----------------------------------------------------------------------------------------------
    // Hybrid mode: use externally supplied mean/variance (as with inference),
    // but also allow
    // autodifferentiation (as with training).
    //----------------------------------------------------------------------------------------------
    if ((exe_mode_ == GraphExeMode::kTrain) && (use_global_stats)) {
      // FIXME: We suspect there's a bug in the gradient calculations performed
      // by this version of
      // nGraph's BatchNorm operator. So for now we'll avoid using it.  -cconvey
      // 2018-04-12.
      // if (ngraph_bn_op_available)
      if (false) {
        const NgraphNodePtr ng_normalized_data =
            std::make_shared<ngraph::op::BatchNorm>(
                eps, ng_actual_gamma, ng_in_beta, ng_in_data, ng_in_moving_mean,
                ng_in_moving_var, true);

        return ng_normalized_data;
      } else {
        // NOTE: This call is intentionally the same as another call below.  The
        // function called
        // here produces a subgraph suitable for training because all of its
        // operators support
        // autodiff.
        const NgraphNodePtr ng_normalized_data =
            create_batchnorm_inference_without_ngraph_bn_op(
                eps, ng_maybe_gamma, ng_in_beta, ng_in_data, ng_in_moving_mean,
                ng_in_moving_var, channel_axis);

        return ng_normalized_data;
      }
    }

    //----------------------------------------------------------------------------------------------
    // Traditional inference mode...
    //----------------------------------------------------------------------------------------------
    if (exe_mode_ == GraphExeMode::kInfer) {
      if (ngraph_bn_op_available) {
        const NgraphNodePtr ng_normalized_data =
            std::make_shared<ngraph::op::BatchNorm>(
                eps, ng_actual_gamma, ng_in_beta, ng_in_data, ng_in_moving_mean,
                ng_in_moving_var, false);

        return ng_normalized_data;
      } else {
        const NgraphNodePtr ng_normalized_data =
            create_batchnorm_inference_without_ngraph_bn_op(
                eps, ng_maybe_gamma, ng_in_beta, ng_in_data, ng_in_moving_mean,
                ng_in_moving_var, channel_axis);

        return ng_normalized_data;
      }
    }

    CHECK(false && "UNEXPECTED: Unhandled BatchNorm mode.");
    return {};
  };

  ngraph_op_funcs_["Convolution"] =
      [this](const NodePtr& node) -> NgraphNodePtr {
    enum InputName { kData = 0, kWeight, kBias };

    NgraphNodePtr data = op_map_[node->inputs_[kData]];
    NgraphNodePtr filter = op_map_[node->inputs_[kWeight]];

    // N, channel_in, d1,...,dn
    const auto data_shape = data->get_shape();
    // channel_out, channel_in/groups, f1,...,fn
    const auto filter_shape = filter->get_shape();

    auto n = data_shape.size() - 2;
    ngraph::CoordinateDiff default_pad(n, 0);
    ngraph::Strides default_stride(n, 1);
    ngraph::Strides default_dilate(n, 1);

    auto pad = get_default<ptrdiff_t>(node, "pad", default_pad);
    auto stride = get_default<size_t>(node, "stride", default_stride);
    auto dilate = get_default<size_t>(node, "dilate", default_dilate);
    size_t groups = get_default(node, "num_group", 1);

    NgraphNodePtr convolution = nullptr;
    if (groups == 1) {
      convolution = std::make_shared<ngraph::op::Convolution>(
          data, filter, stride, dilate, pad, pad);
    } else {
      std::vector<NgraphNodePtr> convolutions(groups);
      for (size_t g = 0; g < groups; ++g) {
        // slice data on channel_in
        size_t data_slice_step = data_shape[1] / groups;
        size_t filter_slice_step = filter_shape[0] / groups;
        auto data_slice = slice_data_on_axis(data, g * data_slice_step,
                                             data_slice_step, 1, false);
        auto filter_slice = slice_data_on_axis(filter, g * filter_slice_step,
                                               filter_slice_step, 0, false);
        // convolve sliced data and filter
        // N, channel_out/groups, d'1,...,d'n
        convolutions[g] = std::make_shared<ngraph::op::Convolution>(
            data_slice, filter_slice, stride, dilate, pad, pad);
      }

      // concatenate convolutions on channel_out
      // N, channel_out, d'1,...,d'n
      convolution = std::make_shared<ngraph::op::Concat>(convolutions, 1);
    }

    // no bias param, return
    if (node->inputs_.size() <= kBias) {
      return convolution;
    }

    NgraphNodePtr bias = op_map_[node->inputs_[kBias]];

    // 1, channel_out, 1,...,1
    ngraph::Shape bias_shape(filter_shape.size(), 1);
    bias_shape[1] = filter_shape[0];

    ngraph::AxisVector order(1, 0);
    auto bias_reshape =
        std::make_shared<ngraph::op::Reshape>(bias, order, bias_shape);

    return ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(
        convolution, bias_reshape);
  };
  ngraph_op_funcs_["Deconvolution"] =
      [this](const NodePtr& node) -> NgraphNodePtr {
    NgraphNodePtr data = op_map_[node->inputs_[0]];
    NgraphNodePtr filter = op_map_[node->inputs_[1]];

    auto conv = create_deconvolution(
        data, filter, TShape_to_NShape(node->shape_), node->orig_node_);

    if (node->inputs_.size() > 2) {
      NgraphNodePtr bias = op_map_[node->inputs_[2]];
      ngraph::Shape bias_shape(filter->get_shape().size(), 1);
      bias_shape[1] = bias->get_shape()[0];

      auto bias_reshape = std::make_shared<ngraph::op::Reshape>(
          bias, ngraph::AxisVector{0}, bias_shape);

      conv = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(
          conv, bias_reshape);
    }
    return conv;
  };

  ngraph_op_funcs_["Pooling"] = [this](const NodePtr& node) {
    return create_pooling(node, op_map_[node->inputs_[0]]);
  };

  ngraph_op_funcs_["SequenceReverse"] =
      [this](const NodePtr& node) -> NgraphNodePtr {
    auto data = op_map_[node->inputs_[0]];
    const bool use_sequence_length =
        get_default(node, "use_sequence_length", false);
    const int seq_axis = get_default(node, "axis", 0);
    if (use_sequence_length) {
      const int batch_axis = 1;
      NgraphNodePtr sequence_length = op_map_[node->inputs_[1]];
      return std::make_shared<ngraph::op::ReverseSequence>(
          data, sequence_length, batch_axis, seq_axis);
    } else {
      return std::make_shared<ngraph::op::Reverse>(
          data, ngraph::AxisSet{static_cast<size_t>(seq_axis)});
    }
  };
  ngraph_op_funcs_["SoftmaxOutput"] = [this](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto in_shape = input->get_shape();

    ngraph::AxisSet axes;
    if (get_default(node, "multi_output", false)) {
      axes.insert(1);
    } else if (get_default(node, "preserve_shape", false)) {
      axes.insert(in_shape.size() - 1);
    } else {
      auto tmpaxes = pyrange(1, in_shape.size());
      axes = std::set<size_t>(tmpaxes.begin(), tmpaxes.end());
    }
    return std::make_shared<ngraph::op::Softmax>(input, axes);
  };
  ngraph_op_funcs_["MakeLoss"] = [this](const NodePtr& node) {
    // MakeLoss forward returns copy/identity
    return op_map_[node->inputs_[0]];
  };
  ngraph_op_funcs_["LinearRegressionOutput"] = [this](const NodePtr& node) {
    return op_map_[node->inputs_[0]];
  };
}

void Emitter::CreateLossOps() {
  // These functions are in place to provide a mechanism for manually specifying
  // backpropgation methods for Loss functions. We do this because MXNet
  // provides a number of options that only effect the output of backprop, not
  // forward prop, and are difficult or impossible to integrate into
  // nGraph's autodiff functionality.
  loss_op_backward_funcs_["SoftmaxOutput"] = [this](
      const NodePtr& node, const NgraphNodePtr& adjoint) {
    const float grad_scale = get_default(node, "grad_scale", 1.0f);
    const float ignore_label = get_default(node, "ignore_label", -1.0f);
    const float smooth_alpha = get_default(node, "smooth_alpha", 0.0f);

    const bool use_ignore = get_default(node, "use_ignore", false);
    const bool out_grad = get_default(node, "out_grad", false);

    const std::string norm =
        get_default(node, "normalization", std::string("null"));

    auto softmax = op_map_[node];
    auto label = op_map_[node->inputs_[1]];
    bool ignore = false;
    NgraphNodePtr mask;

    if (label->get_shape() != softmax->get_shape()) {
      if (use_ignore) {
        ignore = true;
        mask =
            cast_result(std::make_shared<ngraph::op::NotEqual>(
                            label, makeConstant(label->get_element_type(),
                                                label->get_shape(),
                                                std::to_string(ignore_label))),
                        getType(node->dtype_));
      }
      size_t axis = op_map_[node->inputs_[0]]->get_shape().size() - 1;
      if (get_default(node, "multi_output", false)) {
        axis = 1;
      }
      label = std::make_shared<ngraph::op::OneHot>(label, softmax->get_shape(),
                                                   axis);
      if (ignore) {
        // We need to reshape the mast so we can broadcast it with
        // the gradient
        ngraph::Shape new_shape = softmax->get_shape();
        new_shape[axis] = 1;
        mask = std::make_shared<ngraph::op::Reshape>(
            mask, pyrange(mask->get_shape().size()), new_shape);
      }
    }

    if (smooth_alpha != 0.0f) {
      int num_classes = 1;
      auto shape = softmax->get_shape();
      if (get_default(node, "multi_output", false)) {
        num_classes = shape[1];
      } else if (get_default(node, "preserve_shape", false)) {
        num_classes = shape.back();
      } else {
        for (size_t i = 1; i < shape.size(); ++i) {
          num_classes *= shape[i];
        }
      }
      auto one = makeConstant(node, "1");
      auto smooth_const = makeConstant(node, std::to_string(smooth_alpha));
      auto subtractions = label * smooth_const;
      auto additions = (one - label) * smooth_const /
                       makeConstant(node, std::to_string(num_classes - 1));
      label = label - subtractions + additions;
    }

    auto gradient = softmax - label;

    if (ignore) {
      // Mask out the gradient
      gradient =
          ngraph::builder::make_with_numpy_broadcast<ngraph::op::Multiply>(
              gradient, mask);
    }

    if (grad_scale != 1.0f) {
      gradient = gradient * makeConstant(node, std::to_string(grad_scale));
    }

    if (out_grad) {
      gradient = gradient * adjoint;
    }

    if (norm == "batch") {
      gradient =
          gradient / makeConstant(gradient->get_element_type(),
                                  gradient->get_shape(),
                                  std::to_string(gradient->get_shape()[0]));
    } else if (norm == "valid") {
      ngraph::AxisSet axes;
      for (size_t i = 0; i < mask->get_shape().size(); ++i) {
        axes.insert(i);
      }
      gradient = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Divide>(
          gradient, std::make_shared<ngraph::op::Sum>(mask, axes));
    }

    return gradient;
  };
  loss_op_backward_funcs_["MakeLoss"] = [this](const NodePtr& node,
                                               const NgraphNodePtr& adjoint) {
    auto input = op_map_[node->inputs_[0]];
    const std::string norm =
        get_default(node, "normalization", std::string("null"));
    const std::string valid_thresh =
        get_default(node, "valid_thresh", std::string("0"));

    auto grad_scale =
        makeConstant(node, get_default(node, "grad_scale", std::string("1.0")));

    NgraphNodePtr grad;
    if (norm == "valid") {
      auto thresh =
          makeConstant(ngraph::element::f32, input->get_shape(), valid_thresh);
      auto is_gt = std::make_shared<ngraph::op::Greater>(input, thresh);

      auto mask = cast_result(is_gt, input->get_element_type());

      ngraph::AxisSet axes;
      for (auto val : pyrange(mask->get_shape().size())) {
        axes.insert(val);
      }
      NgraphNodePtr sum = std::make_shared<ngraph::op::Sum>(mask, axes);
      NgraphNodePtr one = makeConstant(sum->get_element_type(),
                                       sum->get_shape(), std::string("1"));
      NgraphNodePtr result_norm =
          std::make_shared<ngraph::op::Maximum>(sum, one);

      ngraph::Shape new_shape(grad_scale->get_shape().size(), 1);
      result_norm = std::make_shared<ngraph::op::Reshape>(
          result_norm, pyrange(result_norm->get_shape().size()), new_shape);

      grad = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Divide>(
          grad_scale, result_norm);
    } else if (norm == "batch") {
      grad = grad_scale /
             makeConstant(node, std::to_string(input->get_shape()[0]));
    } else {
      grad = grad_scale;
    }

    return grad;
  };
  loss_op_backward_funcs_["LinearRegressionOutput"] = [this](
      const NodePtr& node, const NgraphNodePtr& adjoint) {
    auto label = op_map_[node->inputs_[0]];
    auto data = op_map_[node->inputs_[1]];
    auto grad_scale =
        makeConstant(node, get_default(node, "grad_scale", std::string("1.0")));
    auto num_output = makeConstant(
        node, std::to_string(node->shape_.Size() / node->shape_[0]));
    return (label - data) * grad_scale / num_output;
  };
}

void Emitter::UnsupportedOps() {
  for (auto kv : ngraph_op_funcs_) {
    supported_ops[kv.first] = [](const NodePtr& node) { return true; };
  }
  supported_ops["BatchNorm"] = [](const NodePtr& node) {
    bool out = true;
    auto shape = TShape_to_NShape(node->inputs_[0]->shape_);
    if (shape[1] % 8 != 0) {
      // MXNet outperforms nGraph in this case.
      out = false;
    }
    return out;
  };
  supported_ops["LeakyReLU"] = [](const NodePtr& node) {
    bool out = true;
    // We haven't yet implemented all activation functions for
    // LeaklyReLU...
    const std::string act_type =
        get_default(node, "act_type", std::string("leaky"));
    if (act_type != "leaky") {
      out = false;
    }
    return out;
  };
  supported_ops["Deconvolution"] = [](const NodePtr& node) {
    // There are some untested arguments to Deconvolution, this is to check and
    // make sure we don't send a deconvolution to nGraph with unsupported
    // arguments
    bool out = true;
    const auto out_shape = TShape_to_NShape(node->shape_);
    auto data = makeConstant(node->inputs_[0], "0");
    auto filter = makeConstant(node->inputs_[1], "0");
    auto conv = create_deconvolution(data, filter, out_shape, node->orig_node_);

    if (conv->get_shape() != TShape_to_NShape(node->shape_)) {
      if (ngraph_log_verbose_detail()) {
        std::cout << "NGRAPH_BRIDGE: Deconvolution with adjust and target "
                     "shape is not tested in MXNet."
                  << std::endl;
        node->printOpDetails(std::cout);
        std::cout << std::endl;
      }
      out = false;
    }
    return out;
  };
}

}  // namespace ngraph_bridge
