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
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LEAKY_RELU_INL_H_
#define MXNET_OPERATOR_LEAKY_RELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../common/random_generator.h"
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./random/sampler.h"
#include "./random/sample_op.h"

namespace mxnet {
namespace op {

namespace leakyrelu {
enum LeakyReLUOpInputs {kData, kGamma};
enum LeakyReLUOpOutputs {kOut, kMask};
enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU, kELU};
enum LeakyReLUOpResource {kRandom};
}  // namespace leakyrelu

struct LeakyReLUParam : public dmlc::Parameter<LeakyReLUParam> {
  // use int for enumeration
  int act_type;
  float slope;
  float lower_bound;
  float upper_bound;
  DMLC_DECLARE_PARAMETER(LeakyReLUParam) {
    DMLC_DECLARE_FIELD(act_type).set_default(leakyrelu::kLeakyReLU)
    .add_enum("rrelu", leakyrelu::kRReLU)
    .add_enum("leaky", leakyrelu::kLeakyReLU)
    .add_enum("prelu", leakyrelu::kPReLU)
    .add_enum("elu", leakyrelu::kELU)
    .describe("Activation function to be applied.");
    DMLC_DECLARE_FIELD(slope).set_default(0.25f)
    .describe("Init slope for the activation. (For leaky and elu only)");
    DMLC_DECLARE_FIELD(lower_bound).set_default(0.125f)
    .describe("Lower bound of random slope. (For rrelu only)");
    DMLC_DECLARE_FIELD(upper_bound).set_default(0.334f)
    .describe("Upper bound of random slope. (For rrelu only)");
  }
};

struct prelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 0.0f : a;
  }
};

template<typename xpu, typename DType>
class LeakyReLUOp : public Operator {
 public:
  explicit LeakyReLUOp(LeakyReLUParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> out;
    Tensor<xpu, 3, DType> mask;
    Tensor<xpu, 1, DType> weight;
    int n = in_data[leakyrelu::kData].shape_[0];
    int k = in_data[leakyrelu::kData].shape_[1];
    Shape<3> dshape = Shape3(n, k, in_data[leakyrelu::kData].Size()/n/k);
    data = in_data[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    out = out_data[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::xelu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, DType>(s);
        if (weight.shape_.Size() == 1) {
          Assign(out, req[leakyrelu::kOut],
                 F<mshadow_op::xelu>(data, mshadow::expr::broadcast_scalar(weight, out.shape_)));
        } else {
          Assign(out, req[leakyrelu::kOut],
                 F<mshadow_op::xelu>(data, mshadow::expr::broadcast<1>(weight, out.shape_)));
        }
        break;
      }
      case leakyrelu::kRReLU: {
        if (ctx.is_train) {
          mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 3, DType>(dshape, s);
          mxnet::op::UniformSampler<xpu> sampler;
          Tensor<xpu, 1, DType> low, high;
          mxnet::op::GetSamplingTempData<xpu, DType>(DType(0.0f), DType(1.0f), ctx, &low, &high);
          mxnet::common::random::RandGenerator<xpu, DType> *pgen =
            ctx.requested[0].get_parallel_random<xpu, DType>();
          Tensor<xpu, 1, DType> out = mask.FlatTo1D();
          sampler.Sample(low, high, out, pgen, s);
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kMask], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::mul, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), mask.dptr_, mask.dptr_,
              DType(param_.upper_bound - param_.lower_bound));
          });
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kMask], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::plus, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), mask.dptr_, mask.dptr_,
              DType(param_.lower_bound));
          });
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::xelu, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), out.dptr_, data.dptr_, mask.dptr_);
          });
        } else {
          const float slope = (param_.lower_bound + param_.upper_bound) / 2.0f;
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::xelu, Req>, xpu>::Launch(
              s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(slope));
          });
        }
        break;
      }
      case leakyrelu::kELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::elu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_,
            DType(param_.slope));
        });
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> output;
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> gdata;
    Tensor<xpu, 3, DType> grad;
    Tensor<xpu, 3, DType> mask;
    Tensor<xpu, 1, DType> weight;
    Tensor<xpu, 1, DType> grad_weight;
    int n = out_grad[leakyrelu::kOut].shape_[0];
    int k = out_grad[leakyrelu::kOut].shape_[1];
    Shape<3> dshape = Shape3(n, k, out_grad[leakyrelu::kOut].Size()/n/k);
    grad = out_grad[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    gdata = in_grad[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    output = out_data[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    if (param_.act_type == leakyrelu::kRReLU) {
      mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    if (param_.act_type == leakyrelu::kPReLU) {
      data = in_data[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mxnet::op::mshadow_op::xelu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              output.dptr_, DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, DType>(s);
        grad_weight = in_grad[leakyrelu::kGamma].get<xpu, 1, DType>(s);
        if (weight.shape_.Size() == 1) {
          Shape<4> gshape = Shape4(1, grad.shape_[0], grad.shape_[1], grad.shape_[2]);
          Assign(grad_weight, req[leakyrelu::kGamma],
                 sumall_except_dim<0>(reshape(F<prelu_grad>(data) * grad, gshape)));
          Assign(gdata, req[leakyrelu::kData],
                 F<mshadow_op::xelu_grad>(data,
                                          mshadow::expr::broadcast_scalar(weight, data.shape_))
                 * grad);
        } else {
          Assign(grad_weight, req[leakyrelu::kGamma],
                 sumall_except_dim<1>(F<prelu_grad>(data) * grad));
          Assign(gdata, req[leakyrelu::kData],
                 F<mshadow_op::xelu_grad>(data, mshadow::expr::broadcast<1>(weight, data.shape_))
                 * grad);
        }
        break;
      }
      case leakyrelu::kRReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, mask) * grad);
        break;
      }
      case leakyrelu::kELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mxnet::op::mshadow_op::elu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              output.dptr_, DType(param_.slope));
        });
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

 private:
  LeakyReLUParam param_;
};  // class LeakyReLUOp

template<typename xpu>
Operator* CreateOp(LeakyReLUParam type, int dtype);

#if DMLC_USE_CXX11
class LeakyReLUProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.act_type == leakyrelu::kPReLU) {
      CHECK_EQ(in_shape->size(), 2U) << "Input:[data, gamma]";
    } else {
      CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    }
    const TShape &dshape = in_shape->at(leakyrelu::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.act_type == leakyrelu::kPReLU) {
      const TShape &gshape = in_shape->at(leakyrelu::kGamma);
      if (gshape.ndim() == 1 && gshape.Size() == 1)
        in_shape->at(leakyrelu::kGamma) = TShape(Shape1(1));
      else
        in_shape->at(leakyrelu::kGamma) = TShape(Shape1(dshape[1]));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    if (param_.act_type == leakyrelu::kRReLU) {
      out_shape->push_back(dshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = -1;
    for (const int& type : *in_type) {
      type_assign(&dtype, type);
    }
    for (const int& type : *out_type) {
      type_assign(&dtype, type);
    }

    for (size_t i = 0; i < in_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, dtype);
    }
    for (size_t i = 0; i < out_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*out_type, i, dtype);
    }
    return dtype != -1;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LeakyReLUProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LeakyReLU";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {out_grad[leakyrelu::kOut],
              out_data[leakyrelu::kOut],
              in_data[leakyrelu::kData],
              in_data[leakyrelu::kGamma]};
    } else if (param_.act_type == leakyrelu::kRReLU) {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kMask], out_data[leakyrelu::kOut]};
    } else {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kData]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[leakyrelu::kOut], in_grad[leakyrelu::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {};
    } else {
      return {{in_data[leakyrelu::kData], out_data[leakyrelu::kOut]}};
    }
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {"data", "gamma"};
    } else {
      return {"data"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {"output", "mask"};
    } else {
      return {"output"};
    }
  }

  int NumOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {ResourceRequest::kRandom};
    } else {
      return std::vector<ResourceRequest>();
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  LeakyReLUParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_LEAKY_RELU_INL_H_

