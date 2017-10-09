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
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_fully_connected-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  // nnp_fully_connected_inference will do optimization for batch-size = 1
  // nnp_fully_connected_output will do optimization for batch-size > 1
  switch (dtype) {
  case mshadow::kFloat32:
    return new NNPACKFullyConnectedOp<cpu, float>(param);
  default:
    break;
  }
#endif
  switch (dtype) {
  case mshadow::kFloat32:
    op = new FullyConnectedOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new FullyConnectedOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *FullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape(1, TShape()), aux_shape;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedProp)
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

If ``flatten`` is set to be true, then the shapes are:

- **data**: `(batch_size, x1, x2, ..., xn)`
- **weight**: `(num_hidden, x1 * x2 * ... * xn)`
- **bias**: `(num_hidden,)`
- **out**: `(batch_size, num_hidden)`

If ``flatten`` is set to be false, then the shapes are:

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(num_hidden, input_dim)`
- **bias**: `(num_hidden,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
