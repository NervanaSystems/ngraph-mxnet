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
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu, Da Zheng
*/

#ifndef MXNET_OPERATOR_NN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpResource {kTempSpace};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU, kSoftSign};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .add_enum("softsign", activation::kSoftSign)
    .describe("Activation function to be applied.");
  }

  bool operator==(const ActivationParam& other) const {
    return this->act_type == other.act_type;
  }
};

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::ActivationParam> {
  size_t operator()(const mxnet::op::ActivationParam& val) {
    return val.act_type;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
void ActivationForward(const OpContext &ctx, const TBlob &in_data,
                       const OpReqType &req, const TBlob &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const size_t sz = in_data.shape_.Size();
  if (sz) {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<ForwardOp, Req>, xpu>::Launch(
        s, sz,
        out_data.dptr<DType>(),
        in_data.dptr<DType>());
    });
  }
}

template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
void ActivationBackward(const OpContext &ctx, const TBlob &out_grad,
                        const TBlob &out_data, const OpReqType &req,
                        const TBlob &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const size_t sz = out_data.shape_.Size();
  if (sz) {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<
        mxnet::op::mxnet_op::backward_grad_tuned<BackwardOp>, Req>, xpu>::Launch(
        s, sz,
        in_grad.dptr<DType>(),
        out_grad.dptr<DType>(),
        out_data.dptr<DType>());
    });
  }
}

template<typename xpu>
void ActivationComputeImpl(const ActivationParam &param, const OpContext &ctx,
                           const TBlob &input, OpReqType req, const TBlob &output) {
  MSHADOW_REAL_TYPE_SWITCH(input.type_flag_, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        ActivationForward<xpu, mshadow_op::relu, mshadow_op::relu_grad, DType>(
            ctx, input, req, output);
        break;
      case activation::kSigmoid:
        ActivationForward<xpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>(
            ctx, input, req, output);
        break;
      case activation::kTanh:
        ActivationForward<xpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>(
            ctx, input, req, output);
        break;
      case activation::kSoftReLU:
        ActivationForward<xpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>(
            ctx, input, req, output);
        break;
      case activation::kSoftSign:
        ActivationForward<xpu, mshadow_op::softsign, mshadow_op::softsign_grad, DType>(
                ctx, input, req, output);
            break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  });
}

template<typename xpu>
void ActivationGradComputeImpl(const ActivationParam &param, const OpContext &ctx,
                               const TBlob &out_grad, const TBlob &out_data,
                               OpReqType req, const TBlob &output) {
  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        ActivationBackward<xpu, mshadow_op::relu, mshadow_op::relu_grad, DType>(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kSigmoid:
        ActivationBackward<xpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kTanh:
        ActivationBackward<xpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kSoftReLU:
        ActivationBackward<xpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kSoftSign:
        ActivationBackward<xpu, mshadow_op::softsign, mshadow_op::softsign_grad, DType>(
                ctx, out_grad, out_data, req, output);
            break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  });
}

template<typename xpu>
void ActivationCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  ActivationComputeImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
}

template<typename xpu>
void ActivationGradCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
#if (MXNET_USE_CUDNN == 1 || MXNET_USE_MKLDNN == 1)
  CHECK_EQ(inputs.size(), 3U);
#else
  CHECK_EQ(inputs.size(), 2U);
#endif
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  ActivationGradComputeImpl<xpu>(param, ctx, inputs[0], inputs[1], req[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_ACTIVATION_INL_H_
