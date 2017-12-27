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
  CreateUnaryOps();
  CreateBinaryOps();
  CreateLayerOps();
}

int get_default(const NodePtr& node, const std::string& key, int default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? std::stoi(node->orig_node_->attrs.dict[key])
             : default_val;
}

inline float get_default(const NodePtr& node, const std::string& key,
                         const float default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? std::stof(node->orig_node_->attrs.dict[key])
             : default_val;
}

bool get_default(const NodePtr& node, const std::string& key,
                 bool default_val) {
  if (node->orig_node_->attrs.dict.count(key)) {
    const std::string& val = node->orig_node_->attrs.dict[key];
    if (val == "True" || val == "1")
      return true;
    else
      return false;
  }
  return default_val;
}

template <typename T>
typename std::enable_if<!std::is_unsigned<T>::value, std::vector<T>>::type
get_default(const NodePtr& node, const std::string& key,
            const std::vector<T>& default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? GetIntVectorFromString<T>(node->orig_node_->attrs.dict[key])
             : default_val;
}

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, std::vector<T>>::type
get_default(const NodePtr& node, const std::string& key,
            const std::vector<T>& default_val) {
  std::vector<T> out;
  if (node->orig_node_->attrs.dict.count(key)) {
    auto tmp = GetIntVectorFromString<int>(node->orig_node_->attrs.dict[key]);
    for (auto val : tmp) {
      if (val >= 0) {
        out.push_back(val);
      } else {
        throw std::string(
            "NGRAPH_BRIDGE: expected unsigned integers but got ") +
            std::to_string(val);
      }
    }
  } else {
    out = default_val;
  }
  return out;
}

/**
 * Transforms input axis attribute with name in key based on MXNet convention (0
 * based index), where
 * negative values means indexing from the right.
 */
inline size_t get_default_transformed_axis(const NodePtr& node,
                                           const std::string& key,
                                           int default_val) {
  const int shape_size = node->shape_.ndim();
  int axis = get_default(node, "axis", default_val);
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

  if (exclude) {
    for (size_t i = 0; i < node->get_shape().size(); ++i)
      if (!in_vec(axes, i)) reduction_axes.insert(i);
  } else {
    for (auto i : axes) reduction_axes.insert(i);
  }

  auto output = func(node, reduction_axes);

  if (keepdims) {
    auto reshape = node->get_shape();
    for (auto i : reduction_axes) reshape[i] = 1;

    ngraph::AxisVector order(output->get_shape().size());
    std::iota(order.begin(), order.end(), 0);

    output = std::make_shared<ngraph::op::Reshape>(output, order, reshape);
  }

  return output;
}

NgraphNodePtr Emitter::ReduceAxes(
    const NodePtr& node,
    const std::function<NgraphNodePtr(const NgraphNodePtr&,
                                      const ngraph::AxisSet&)>& func) {
  auto input = op_map_[node->inputs_[0]];
  ngraph::AxisVector axes_numbers(input->get_shape().size());
  std::iota(axes_numbers.begin(), axes_numbers.end(), 0);
  return ReduceAxes(input, get_default(node, "axis", axes_numbers),
                    get_default(node, "exclude", false),
                    get_default(node, "keepdims", false), func);
}

// unary op function generator
void Emitter::CreateUnaryOps() {
  ngraph_op_funcs_["relu"] = [this](const NodePtr& node) {
    auto zero = makeConstant(node, "0");
    return std::make_shared<ngraph::op::Maximum>(op_map_[node->inputs_[0]],
                                                 zero);
  };
  ngraph_op_funcs_["sigmoid"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    return (one / (one + std::make_shared<ngraph::op::Exp>(
                             -op_map_[node->inputs_[0]])));
  };
  // ngraph_op_funcs_["softmax"] = [this](const NodePtr& node) {
  //   auto numer =
  //   std::make_shared<ngraph::op::Exp>(op_map_[node->inputs_[0]]); auto denom
  //   = std::make_shared<ngraph::op::Sum>(numer, ngraph::AxisSet{1}); return ;
  // };
  // ngraph_op_funcs_["log_softmax"] = [this](const NodePtr& node){
  //   return ;
  // };
  ngraph_op_funcs_["_copy"] = [this](const NodePtr& node) {
    return op_map_[node->inputs_[0]];  // TODO: Return this as a reference. Does
                                       // it actually need to be copied?
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
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]], two);
  };
  ngraph_op_funcs_["sqrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]],
                                               one / two);
  };
  ngraph_op_funcs_["rsqrt"] = [this](const NodePtr& node) {
    auto one = makeConstant(node, "1");
    auto two = makeConstant(node, "2");
    return one / std::make_shared<ngraph::op::Power>(op_map_[node->inputs_[0]],
                                                     one / two);
  };
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
  // ngraph_op_funcs_["arcsinh"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["arccosh"] = [this](const NodePtr& node){
  //   return ;
  // };
  // ngraph_op_funcs_["arctanh"] = [this](const NodePtr& node){
  //   return ;
  // };

  ngraph_op_funcs_["_zeros"] = [this](const NodePtr& node) {
    return makeConstant(node, "0");
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
  ngraph_op_funcs_["reshape"] = [this](const NodePtr& node) {
    auto new_shape = TShape_to_NShape(node->shape_);

    auto input = op_map_[node->inputs_[0]];
    if (new_shape.size() ==
        0)  // ngraph++'s reshape wouldn't like an empty shape
    {
      // std::shared_ptr<ngraph::Node> is needed to reconciale
      // ngraph::op::Constant and ngraph::op::Reshape return types
      return std::shared_ptr<ngraph::Node>(
          std::make_shared<ngraph::op::Constant>(input->get_element_type(),
                                                 ngraph::Shape{}, "0"));
    }

    ngraph::AxisVector order(input->get_shape().size());
    std::iota(begin(order), end(order), 0);
    return std::shared_ptr<ngraph::Node>(
        std::make_shared<ngraph::op::Reshape>(input, order, new_shape));
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
  ngraph_op_funcs_["_mod"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Remainder>(op_map_[node->inputs_[0]],
                                                   op_map_[node->inputs_[1]]);
  };
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
  ngraph_op_funcs_["_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Equal>(op_map_[node->inputs_[0]],
                                               op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_not_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::NotEqual>(op_map_[node->inputs_[0]],
                                                  op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_greater"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Greater>(op_map_[node->inputs_[0]],
                                                 op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_greater_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::GreaterEq>(op_map_[node->inputs_[0]],
                                                   op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_lesser"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::Less>(op_map_[node->inputs_[0]],
                                              op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["_lesser_equal"] = [this](const NodePtr& node) {
    return std::make_shared<ngraph::op::LessEq>(op_map_[node->inputs_[0]],
                                                op_map_[node->inputs_[1]]);
  };
  ngraph_op_funcs_["dot"] = [this](const NodePtr& node) {
    NgraphNodePtr left = op_map_[node->inputs_[0]];
    NgraphNodePtr right = op_map_[node->inputs_[1]];
    if (get_default(node, "transpose_a", false)) {
      auto N = left->get_shape().size();
      ngraph::AxisVector order(N - 1);
      std::iota(order.begin(), order.end(), 1);
      order.push_back(0);
      left = ngraph::builder::numpy_transpose(left, order);
    }
    if (get_default(node, "transpose_b", false)) {
      auto N = right->get_shape().size();
      ngraph::AxisVector order(N - 1);
      std::iota(order.begin(), order.end(), 0);
      order.insert(order.begin(), N - 1);
      right = ngraph::builder::numpy_transpose(right, order);
    }
    return std::make_shared<ngraph::op::Dot>(left, right, 1);
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
  ngraph_op_funcs_["broadcast_mod"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Remainder>(node);
  };
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
  ngraph_op_funcs_["broadcast_equal"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Equal>(node);
  };
  ngraph_op_funcs_["broadcast_not_equal"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::NotEqual>(node);
  };
  ngraph_op_funcs_["broadcast_greater"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Greater>(node);
  };
  ngraph_op_funcs_["broadcast_greater_equal"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::GreaterEq>(node);
  };
  ngraph_op_funcs_["broadcast_lesser"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::Less>(node);
  };
  ngraph_op_funcs_["broadcast_lesser_equal"] = [this](const NodePtr& node) {
    return CreateAutoBroadcast<ngraph::op::LessEq>(node);
  };
}

// MXNet high level ops generating function
void Emitter::CreateLayerOps() {
  // In mxnet, split takes a tensor and creates multiple tensors from
  // equal slices along 1 axis. The compiler creates a subgraph where
  // each of those outputs is a single node.  This function creates
  // the slice op for making each tensor.
  ngraph_op_funcs_["split"] = [this](const NodePtr& node) {
    size_t axis = get_default(node, "axis", 1);
    int num_outputs = get_default(node, "num_outputs", 1);
    int index = node->multi_output_index_;
    bool squeeze_axis = get_default(node, "squeeze_axis", 0);

    // create lower and upper bounds for slice
    auto upper = TShape_to_NShape(node->inputs_[0]->shape_);
    std::vector<size_t> lower(upper.size(), 0);

    lower[axis] = index * upper[axis] / num_outputs;
    upper[axis] = (index + 1) * upper[axis] / num_outputs;

    // perform the slice
    std::shared_ptr<ngraph::Node> op = std::make_shared<ngraph::op::Slice>(
        op_map_[node->inputs_[0]], lower, upper);

    // remove dimension 1 axis if needed
    if (squeeze_axis && ((upper[axis] - lower[axis]) == 1)) {
      std::vector<size_t> reshape;
      for (size_t i = 0; i < upper.size(); ++i)
        if (i != axis) reshape.push_back(upper[i]);

      // can this be a reshape default?
      ngraph::AxisVector order(upper.size());
      std::iota(order.begin(), order.end(), 0);

      op = std::make_shared<ngraph::op::Reshape>(op, order, reshape);
    }

    return op;
  };

  // concat takes a list of tensors of equal shape and
  // concatenates them along a given axis
  ngraph_op_funcs_["concat"] = [this](const NodePtr& node) {
    // get the concat axis
    size_t axis = get_default(node, "dim", 1);

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
    auto beta = op_map_[node->inputs_[2]];
    auto dot = std::make_shared<ngraph::op::Dot>(
        X, ngraph::builder::numpy_transpose(W));

    return ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(dot,
                                                                       beta);
  };

  // flatten converts an array of shape (x0, x1, x2, ...)
  // to an array of shape (x0, x1*x2*...)
  ngraph_op_funcs_["flatten"] = [this](const NodePtr& node) {
    auto in_shape = TShape_to_NShape(node->inputs_[0]->shape_);
    auto out_shape = ngraph::Shape({in_shape[0], 1});
    out_shape[1] = std::accumulate(in_shape.begin() + 1, in_shape.end(), 1,
                                   std::multiplies<int>());
    // Create a range vector indicating that
    // Reshape should take the axes in order
    // these two lines are use all over the place
    ngraph::AxisVector order(in_shape.size());
    std::iota(order.begin(), order.end(), 0);

    return std::make_shared<ngraph::op::Reshape>(op_map_[node->inputs_[0]],
                                                 order, out_shape);
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

    // Create a range vector indicating that
    // Reshape should take the axes in order
    ngraph::AxisVector order(in_shape.size());
    std::iota(order.begin(), order.end(), 0);

    // copy the shape and insert a 1 at the axis position to expand the
    // dimension
    auto out_shape = in_shape;
    out_shape.insert(out_shape.begin() + axis, 1);

    return std::make_shared<ngraph::op::Reshape>(op_map_[node->inputs_[0]],
                                                 order, out_shape);
  };

  // batch norm operation
  ngraph_op_funcs_["BatchNorm"] = [this](const NodePtr& node) {
    // TODO lfeng:
    // - support use_global_stats (moving_mean & moving_variance), this feature
    // requires multiple outputs.

    enum InputName { kData = 0, kGamma, kBeta, kMovingMean, kMovingVar };
    NgraphNodePtr ng_in_data = op_map_[node->inputs_[kData]];
    NgraphNodePtr ng_in_gamma = op_map_[node->inputs_[kGamma]];
    NgraphNodePtr ng_in_beta = op_map_[node->inputs_[kBeta]];
    const int data_shape_size =
        static_cast<int>(ng_in_data->get_shape().size());

    // Default Batch norm parameters
    const float eps = get_default(node, "eps", 0.001f);
    const float momentum = get_default(node, "momentum", 0.9f);
    const bool fix_gamma = get_default(node, "fix_gamma", true);
    const bool use_global_stats = get_default(node, "use_global_stats", false);
    // zero based channel axis
    const size_t channel_axis = get_default_transformed_axis(node, "axis", 1);

    NgraphNodePtr ng_mean = ReduceAxes(ng_in_data, {channel_axis}, true, true,
                                       ngraph::builder::mean);
    NgraphNodePtr ng_var =
        ReduceAxes(ng_in_data, {channel_axis}, true, true,
                   [](const std::shared_ptr<ngraph::Node>& node,
                      const ngraph::AxisSet& axes) {
                     return ngraph::builder::variance(node, axes);
                   });

    using ngraph::builder::make_with_numpy_broadcast;

    NgraphNodePtr ng_eps = makeConstant(node, std::to_string(eps));
    NgraphNodePtr denom = std::make_shared<ngraph::op::Sqrt>(
        make_with_numpy_broadcast<ngraph::op::Add>(ng_var, ng_eps));

    NgraphNodePtr numerator =
        make_with_numpy_broadcast<ngraph::op::Subtract>(ng_in_data, ng_mean);

    NgraphNodePtr result =
        make_with_numpy_broadcast<ngraph::op::Divide>(numerator, denom);

    // we need to convert gamma and beta to proper shape similar to mean and
    // variance thought ReduceAxes. Gamma and beta should already have shape
    // like [1, C], we want to make sure it's properly shape to [C, 1] depending
    // on the index of channel.
    ngraph::AxisVector convert_order(ng_in_gamma->get_shape().size());
    std::iota(begin(convert_order), end(convert_order), 0);
    // fill the shape with (shape_size - 1) of 1s.
    ngraph::Shape convert_shape(data_shape_size - 1, 1);
    // number of elements for channel axis
    size_t channel_size = ng_in_data->get_shape()[channel_axis];
    // insert channel size at the proper index for the channel
    convert_shape.insert(convert_shape.begin() + channel_axis, channel_size);

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
}

}  // namespace ngraph_bridge
