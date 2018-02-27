/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);

inline std::string get_reduce_axes_description(const std::string& op_name, int line) {
  std::string doc = R"code(Computes the __op__ of array elements over given axes.

Defined in )code";
  doc += std::string(__FILE__) + std::string(":L") + std::to_string(line);
  size_t pos = 0;
  std::string holder("__op__");
  while ((pos = doc.find(holder, pos)) != std::string::npos) {
    doc.replace(pos, holder.length(), op_name);
    pos += op_name.length();
  }
  return doc;
}

MXNET_OPERATOR_REGISTER_REDUCE(sum)
MXNET_ADD_SPARSE_OP_ALIAS(sum)
.add_alias("sum_axis")
.describe(R"code(Computes the sum of array elements over given axes.

.. Note::

  `sum` and `sum_axis` are equivalent.
  For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  Setting keepdims or exclude to True will cause a fallback to dense operator.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

  data = [[1,2,0],
          [3,0,1],
          [4,1,0]]

  csr = cast_storage(data, 'csr')

  sum(csr, axis=0)
  [ 8.  3.  1.]

  sum(csr, axis=1)
  [ 3.  4.  5.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SumOpForwardEx<cpu, mshadow::red::sum>)
.set_attr<FInferStorageType>("FInferStorageType", SumOpForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_sum)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu>);

MXNET_OPERATOR_REGISTER_REDUCE(mean)
MXNET_ADD_SPARSE_OP_ALIAS(mean)
.describe(get_reduce_axes_description("mean", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum, true>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SumOpForwardEx<cpu, mshadow::red::sum, true>)
.set_attr<FInferStorageType>("FInferStorageType", SumOpForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mean"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_mean)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu, true>);

MXNET_OPERATOR_REGISTER_REDUCE(prod)
.describe(get_reduce_axes_description("product", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::product>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_prod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_prod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut< cpu, mshadow_op::rdiv>);

MXNET_OPERATOR_REGISTER_REDUCE(nansum)
.describe(R"code(Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nansum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nansum" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nansum)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nansum_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(nanprod)
.describe(R"code(Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nanprod>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nanprod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nanprod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nanprod_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(max)
.add_alias("max_axis")
.describe(get_reduce_axes_description("max", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::maximum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_max"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_max)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_REDUCE(min)
.add_alias("min_axis")
.describe(get_reduce_axes_description("min", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::minimum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_min"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_min)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_axis)
.add_alias("broadcast_axes")
.describe(R"code(Broadcasts the input array over particular axes.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

Example::

   // given x of shape (1,2,1)
   x = [[[ 1.],
         [ 2.]]]

   // broadcast x on on axis 2
   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                         [ 2.,  2.,  2.]]]
   // broadcast x on on axes 0 and 2
   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]],
                                                [[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastAxesParam>)
.add_arguments(BroadcastAxesParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_to)
.describe(R"code(Broadcasts the input array to a new shape.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimension which you do not want to change can also be kept as `0` which means copy the original value.
So with `shape=(2,0)`, we will obtain the same result as in the above example.

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastToParam>)
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

// backward op for broadcast.
NNVM_REGISTER_OP(_broadcast_backward)
.set_attr_parser(ParamParser<ReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

NNVM_REGISTER_OP(norm)
MXNET_ADD_SPARSE_OP_ALIAS(norm)
.describe(R"code(Flattens the input array and then computes the l2 norm.

Examples::

  x = [[1, 2],
       [3, 4]]

  norm(x) = [5.47722578]

  rsp = x.cast_storage('row_sparse')

  norm(rsp) = [5.47722578]

  csr = x.cast_storage('csr')

  norm(csr) = [5.47722578]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<TShape> *in_attrs,
     std::vector<TShape> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    if ((*in_attrs)[0].ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", L2NormStorageType)
.set_attr<FCompute>("FCompute<cpu>", L2NormCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", L2NormComputeEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Source input");

}  // namespace op
}  // namespace mxnet
