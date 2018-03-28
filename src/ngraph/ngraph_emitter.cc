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
#include "ngraph_utils.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/op/get_output_element.hpp>
#include "ngraph_sgcompiler_utils.h"

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
}

void Emitter::ClearOpMap() {
  // delete the temporary storage
  op_map_.clear();
  placeholder_order_.clear();
}

/**
 * Transforms input axis attribute with name in key based on MXNet convention (0
 * based index), where
 * negative values means indexing from the right.
 */
inline size_t get_default_transformed_axis(const NodePtr& node,
                                           const std::string& key,
                                           const int default_val,
                                           const int shape_size) {
  int axis = get_default(node, key, default_val);
  assert(abs(axis) <= shape_size);
  // convert negative axis index to postive (counting from right per mxnet
  // convention)
  size_t transformed_axis = axis < 0 ? shape_size + axis : axis;

  return transformed_axis;
}

/**
 * Performs a reduction on the node using the input axes.
 * @param axes list of axis to operate on
 * @param exclude if true should use node axes not listed in axes parameter,
 *  otherwise use input axes for the reduction operation.
 * @param keepdims if true the result will be reshaped to have the same
 *  dimention as the node (where reduction axes will have size 1), otherwise
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
  if (axes.size() == 0) {
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
                                      const ngraph::AxisSet&)>& func) {
  auto input = op_map_[node->inputs_[0]];
  return ReduceAxes(
      input, get_default(node, "axis", pyrange(input->get_shape().size())),
      get_default(node, "exclude", false), get_default(node, "keepdims", false),
      func);
}

// unary op function generator
void Emitter::CreateUnaryOps() {
  ngraph_op_funcs_["Activation"] = [this](const NodePtr node) {
    auto act_type = node->orig_node_->attrs.dict["act_type"];
    return ngraph_op_funcs_[act_type](node);
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
  // TODO(mbrookhart): Arc trig autodiff not implemented
  /*
  ngraph_op_funcs_["arcsin"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Asin>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["arccos"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Acos>(op_map_[node->inputs_[0]]);
  };
  ngraph_op_funcs_["arctan"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Atan>(op_map_[node->inputs_[0]]);
  };
  */
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

  //----------------------------- Reduce Ops ----------------------------//
  ngraph_op_funcs_["norm"] = [this](const NodePtr& node) {
    return ReduceAxes(node, ngraph::builder::l2_norm);
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
  auto arg1 =
      makeConstant(node, std::to_string(get_default(node, "scalar", 0.0f)));
  return ngraph::builder::make_with_numpy_broadcast<op>(arg0, arg1);
}
// cast result of op to given type
NgraphNodePtr cast_result(const NgraphNodePtr& op,
                          const ngraph::element::Type& type) {
  return std::make_shared<ngraph::op::Convert>(op, type);
}

NgraphNodePtr slice_data_on_axis(NgraphNodePtr data, size_t starting_loc,
                                 size_t step_size = 1, size_t axis = 0,
                                 bool flatten = true) {
  // slice data on given axis
  ngraph::Coordinate lower(data->get_shape().size(), 0);
  ngraph::Coordinate upper = data->get_shape();

  lower[axis] = starting_loc;
  upper[axis] = starting_loc + step_size;

  NgraphNodePtr slice = std::make_shared<ngraph::op::Slice>(data, lower, upper);

  if (flatten && (step_size == 1)) {
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < slice->get_shape().size(); ++i) {
      if (i != axis) {
        out_shape.push_back(slice->get_shape()[i]);
      }
    }
    slice = std::make_shared<ngraph::op::Reshape>(
        slice, pyrange(data->get_shape().size()), out_shape);
  }

  return slice;
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
    return std::make_shared<ngraph::op::Dot>(args.first, args.second, 1);
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
  ngraph_op_funcs_["_add_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Add>(node);
  };
  ngraph_op_funcs_["_minus_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Subtract>(node);
  };
  ngraph_op_funcs_["_mul_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Multiply>(node);
  };
  ngraph_op_funcs_["_div_scalar"] = [this](const NodePtr& node) {
    return CreateScalarOp<ngraph::op::Divide>(node);
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
      auto mask_shape = TShape_to_NShape(node->shape_);
      auto mask = ngraph::builder::tensor_mask(sequence_lengths, sequence_axis,
                                               batch_axis, mask_shape);

      // create value constant, same shape as node
      auto value = get_default(node, "value", std::string("0"));
      auto value_constant =
          makeConstant(ngraph::element::f32, mask_shape, value);

      // data[mask==False] = value
      data = std::make_shared<ngraph::op::Select>(mask, data, value_constant);
    }
    return data;
  };
}

struct PoolingParams {
  PoolingParams(const NodePtr& node, const NgraphNodePtr& input) {
    pooling_convention =
        get_default(node, "pooling_convention", std::string("valid"));
    global_pool = get_default(node, "global_pool", false);

    auto input_shape = input->get_shape();
    // first two tensor axes are batch and channel, rest are image channels
    // get the number of image channels for pooling
    auto pool_dim = input_shape.size() - 2;
    auto default_ones = std::vector<size_t>(pool_dim, 1);
    auto default_zeros = std::vector<size_t>(pool_dim, 0);

    kernel = get_default(node, "kernel", default_ones);
    stride = get_default(node, "stride", default_ones);
    pad = get_default(node, "pad", default_zeros);

    // if global pooling is true, reset the pooling kernel to the
    // input image size
    if (global_pool) {
      kernel = std::vector<size_t>();
      // get all of the image dimensions for kernel
      for (size_t i = 2; i < input_shape.size(); ++i) {
        kernel.push_back(input_shape[i]);
      }
    }
  }

  std::string pooling_convention;
  bool global_pool;
  std::vector<size_t> kernel;
  std::vector<size_t> stride;
  std::vector<size_t> pad;
};

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
      dot = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(dot,
                                                                        beta);
    }
    return dot;
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

    NgraphNodePtr ng_mean{nullptr};
    NgraphNodePtr ng_var{nullptr};

    using ngraph::builder::make_with_numpy_broadcast;

    // we need to convert some of the input data to proper shape similar to mean
    // and variance through ReduceAxes. They should already have shape
    // like [1, C], we want to make sure it's properly shape to [C, 1] depending
    // on the index of channel.
    auto convert_order = pyrange(ng_in_gamma->get_shape().size());
    // fill the shape with (shape_size - 1) of 1s.
    ngraph::Shape convert_shape(data_shape_size - 1, 1);
    // number of elements for channel axis
    size_t channel_size = ng_in_data->get_shape()[channel_axis];
    // insert channel size at the proper index for the channel
    convert_shape.insert(convert_shape.begin() + channel_axis, channel_size);

    if (data_shape_size == 4 && channel_axis == 1 &&
        node->dtype_ == mshadow::kFloat32 &&
        exe_mode_ == GraphExeMode::kTrain && !use_global_stats) {
      NgraphNodePtr gamma;
      if (fix_gamma) {
        gamma = makeConstant(ng_in_moving_mean->get_element_type(),
                             ng_in_moving_mean->get_shape(), "1");
      } else {
        gamma = ng_in_gamma;
      }
      auto BN = std::make_shared<ngraph::op::BatchNorm>(eps, gamma, ng_in_beta,
                                                        ng_in_data);
      ng_mean = std::make_shared<ngraph::op::GetOutputElement>(BN, 1);
      ng_var = std::make_shared<ngraph::op::GetOutputElement>(BN, 2);

      NgraphNodePtr ng_one = makeConstant(ng_in_moving_mean->get_element_type(),
                                          ng_in_moving_mean->get_shape(), "1");
      NgraphNodePtr ng_momentum =
          makeConstant(ng_in_moving_var->get_element_type(),
                       ng_in_moving_var->get_shape(), std::to_string(momentum));

      aux_op_map_[node->inputs_[kMovingMean]] =
          ng_in_moving_mean * ng_momentum + ng_mean * (ng_one - ng_momentum);
      aux_op_map_[node->inputs_[kMovingVar]] =
          ng_in_moving_var * ng_momentum + ng_var * (ng_one - ng_momentum);
      return std::make_shared<ngraph::op::GetOutputElement>(BN, 0);
    }

    if (exe_mode_ == GraphExeMode::kTrain && !use_global_stats) {
      ng_mean = ReduceAxes(ng_in_data, {channel_axis}, true, true,
                           ngraph::builder::mean);
      ng_var = ReduceAxes(ng_in_data, {channel_axis}, true, true,
                          [](const std::shared_ptr<ngraph::Node>& node,
                             const ngraph::AxisSet& axes) {
                            return ngraph::builder::variance(node, axes);
                          });
      ngraph::AxisVector order(ng_mean->get_shape().size());
      std::iota(order.begin(), order.end(), 0);
      auto ng_mean_temp = std::make_shared<ngraph::op::Reshape>(
          ng_mean, order, ng_in_moving_mean->get_shape());
      auto ng_var_temp = std::make_shared<ngraph::op::Reshape>(
          ng_var, order, ng_in_moving_var->get_shape());

      // update running averages

      NgraphNodePtr ng_one = makeConstant(ng_in_moving_mean->get_element_type(),
                                          ng_in_moving_mean->get_shape(), "1");
      NgraphNodePtr ng_momentum =
          makeConstant(ng_in_moving_var->get_element_type(),
                       ng_in_moving_var->get_shape(), std::to_string(momentum));
      ngraph::Shape s = ng_in_moving_mean->get_shape();

      aux_op_map_[node->inputs_[kMovingMean]] =
          ng_in_moving_mean * ng_momentum +
          ng_mean_temp * (ng_one - ng_momentum);
      aux_op_map_[node->inputs_[kMovingVar]] =
          ng_in_moving_var * ng_momentum + ng_var_temp * (ng_one - ng_momentum);

    } else {
      // we expect to use global stats with inference
      ng_mean = std::make_shared<ngraph::op::Reshape>(
          ng_in_moving_mean, convert_order, convert_shape);
      ng_var = std::make_shared<ngraph::op::Reshape>(
          ng_in_moving_var, convert_order, convert_shape);
    }

    NgraphNodePtr ng_eps = makeConstant(
        ng_var->get_element_type(), ng_var->get_shape(), std::to_string(eps));
    NgraphNodePtr denom = std::make_shared<ngraph::op::Sqrt>(ng_var + ng_eps);

    NgraphNodePtr numerator =
        make_with_numpy_broadcast<ngraph::op::Subtract>(ng_in_data, ng_mean);

    NgraphNodePtr result =
        make_with_numpy_broadcast<ngraph::op::Divide>(numerator, denom);

    ng_in_gamma = std::make_shared<ngraph::op::Reshape>(
        ng_in_gamma, convert_order, convert_shape);
    ng_in_beta = std::make_shared<ngraph::op::Reshape>(
        ng_in_beta, convert_order, convert_shape);

    // If fix_gamma is true, we assume it to be 1, otherwise, we need to scale
    // result with gamma
    if (!fix_gamma) {
      result =
          make_with_numpy_broadcast<ngraph::op::Multiply>(result, ng_in_gamma);
    }
    result = make_with_numpy_broadcast<ngraph::op::Add>(result, ng_in_beta);
    return result;
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
        size_t slice_step = data_shape[1] / groups;
        auto data_slice =
            slice_data_on_axis(data, g * slice_step, slice_step, 1, false);
        auto filter_slice =
            slice_data_on_axis(filter, g * slice_step, slice_step, 0, false);

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
  ngraph_op_funcs_["Pooling"] = [this](const NodePtr& node) -> NgraphNodePtr {
    NgraphNodePtr op;
    std::string type = get_default(node, "pool_type", std::string("max"));
    if (type == "max") {
      op = ngraph_op_funcs_["max_pooling"](node);
    } else if (type == "avg") {
      op = ngraph_op_funcs_["avg_pooling"](node);
    } else if (type == "sum") {
      op = ngraph_op_funcs_["sum_pooling"](node);
    }
    return op;
  };

  auto asymetric_padding = [](ngraph::Shape input_shape, PoolingParams params) {
    auto top_pad = params.pad;
    if (params.pooling_convention == "full") {
      for (size_t i = 2; i < input_shape.size(); ++i) {
        size_t padded_dim = input_shape[i] + 2 * top_pad[i - 2];
        size_t stride = params.stride[i - 2];
        // calculate extra padding
        auto num_strides = static_cast<size_t>(
            ceil(static_cast<float>(padded_dim - params.kernel[i - 2]) /
                 static_cast<float>(stride)));
        size_t extra_pad =
            num_strides * stride + params.kernel[i - 2] - padded_dim;
        top_pad[i - 2] += extra_pad;
      }
    }
    return top_pad;
  };

  ngraph_op_funcs_["max_pooling"] = [this,
                                     &asymetric_padding](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto params = PoolingParams(node, input);

    return std::make_shared<ngraph::op::MaxPool>(
        input, params.kernel, params.stride, params.pad,
        asymetric_padding(input->get_shape(), params));
  };
  ngraph_op_funcs_["avg_pooling"] = [this,
                                     &asymetric_padding](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto params = PoolingParams(node, input);

    return std::make_shared<ngraph::op::AvgPool>(
        input, params.kernel, params.stride, params.pad,
        asymetric_padding(input->get_shape(), params), true);
  };
  ngraph_op_funcs_["sum_pooling"] = [this,
                                     &asymetric_padding](const NodePtr& node) {
    auto input = op_map_[node->inputs_[0]];
    auto params = PoolingParams(node, input);

    // Compute the sum-pool by first computing the avg-pool, and then
    // element-wise multiply (the resulting vector by each element of the
    // resulting tensor) with (the number of elements in the pooling window).
    // We do this because nGraph++ doesn't directly support sum-pooling.

    const size_t num_window_elements = ngraph::shape_size(params.kernel);

    const auto avg_pool_op = std::make_shared<ngraph::op::AvgPool>(
        input, params.kernel, params.stride, params.pad,
        asymetric_padding(input->get_shape(), params), true);

    const auto coeff_op = ngraph_bridge::makeConstant(
        avg_pool_op->get_element_type(), avg_pool_op->get_shape(),
        std::to_string(num_window_elements));

    auto mul_op = std::make_shared<ngraph::op::Multiply>(avg_pool_op, coeff_op);

    return mul_op;
  };
}

}  // namespace ngraph_bridge
