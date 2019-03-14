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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fc.cc
 * \brief MKLDNN (Quantized) FullyConnected operator based on subgraph
 * \author Ciyong Chen
*/

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <string>
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCOp {
 public:
  explicit SgMKLDNNFCOp(const nnvm::NodeAttrs &attrs)
    : initialized_(false),
      subgraph_sym_(*attrs.subgraphs[0]),
      full_param_(nnvm::get<MKLDNNFCFullParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_;
  nnvm::Symbol subgraph_sym_;
  MKLDNNFCFullParam full_param_;
  std::shared_ptr<MKLDNNFullyConnectedForward> fwd_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_min_data_;
  float cached_max_data_;
  float cached_min_weight_;
  float cached_max_weight_;
  float cached_min_bias_;
  float cached_max_bias_;
};

void SgMKLDNNFCOp::Forward(const OpContext &ctx,
                           const std::vector<NDArray> &in_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &out_data) {
  auto &mkldnn_param = full_param_.mkldnn_param;
  auto &default_param = full_param_.default_param;
  bool has_bias = !default_param.no_bias;
  size_t base_num_inputs = has_bias ? 3 : 2;
  size_t total_num_inputs = base_num_inputs;
  size_t base_num_outputs = 1;
  size_t total_num_outputs = base_num_outputs;

  float min_data = 0.0;
  float max_data = 0.0;
  float min_weight = 0.0;
  float max_weight = 0.0;
  float min_bias = 0.0;
  float max_bias = 0.0;
  float *min_output_ptr = nullptr;
  float *max_output_ptr = nullptr;

  if (mkldnn_param.quantized) {
    total_num_inputs = base_num_inputs * 3;
    min_data = in_data[base_num_inputs].data().dptr<float>()[0];
    max_data = in_data[base_num_inputs + 1].data().dptr<float>()[0];
    min_weight = in_data[base_num_inputs + 2].data().dptr<float>()[0];
    max_weight = in_data[base_num_inputs + 3].data().dptr<float>()[0];
    if (has_bias) {
      min_bias = in_data[base_num_inputs + 4].data().dptr<float>()[0];
      max_bias = in_data[base_num_inputs + 5].data().dptr<float>()[0];
    }
    if (!mkldnn_param.enable_float_output) {
      total_num_outputs = base_num_outputs * 3;
      min_output_ptr = out_data[1].data().dptr<float>();
      max_output_ptr = out_data[2].data().dptr<float>();
    }
  }
  CHECK_EQ(in_data.size(), total_num_inputs);
  CHECK_EQ(out_data.size(), total_num_outputs);

  NDArray data = in_data[fullc::kData];
  NDArray weight = in_data[fullc::kWeight];
  NDArray output = out_data[fullc::kOut];
  const mxnet::TShape &ishape = data.shape();
  if (mkldnn_param.quantized && ishape.ndim() != 2) {
    CHECK(default_param.flatten)
      << "QuantizedFullyConnected only supports flatten=true when ishape.ndim() != 2 for now.";
  }

  mkldnn::memory::desc out_md = GetMemDesc(output);
  MKLDNNFCFlattenData(default_param, out_data[fullc::kOut], &data, &out_md);

  if (initialized_ && mkldnn_param.quantized) {
    if (cached_min_data_ != min_data || cached_max_data_ != max_data ||
        cached_min_weight_ != min_weight || cached_max_weight_ != max_weight ||
        (has_bias && (cached_min_bias_ != min_bias || cached_max_bias_ != max_bias))) {
          initialized_ = false;
        }
  }

  if (!initialized_) {
    cached_min_data_ = min_data;
    cached_max_data_ = max_data;
    cached_min_weight_ = min_weight;
    cached_max_weight_ = max_weight;
    if (has_bias) {
      cached_bias_ = in_data[fullc::kBias];
    } else {
      cached_bias_ = NDArray();
    }

    if (mkldnn_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
      float data_scale  = data_range / MaxAbs(cached_min_data_, cached_max_data_);
      float weight_scale = kInt8Range / MaxAbs(cached_min_weight_, cached_max_weight_);
      float quantized_out_range = mkldnn_param.with_relu ? kUint8Range : kInt8Range;

      if (has_bias) {
        NDArray bias = in_data[fullc::kBias];
        float bias_int32_rescale = data_scale * weight_scale *
            MaxAbs(min_bias, max_bias) / kInt8Range;

        cached_bias_ = NDArray(bias.storage_type(), bias.shape(),
                               bias.ctx(), true, mshadow::kInt32);
        int8_t *bias_ptr = bias.data().dptr<int8_t>();
        int32_t *quantized_bias_ptr = cached_bias_.data().dptr<int32_t>();
        size_t bias_size = bias.shape().Size();
        #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
        for (size_t i = 0; i < bias_size; ++i) {
          quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
        }
      }

      if (mkldnn_param.enable_float_output) {
        full_param_.output_scales[0] = 1.0 / data_scale / weight_scale;
        full_param_.requantize_scales.resize(0);
      } else if (mkldnn_param.min_calib_range.has_value() &&
                 mkldnn_param.max_calib_range.has_value()) {
        full_param_.output_scales.resize(0);
        *min_output_ptr = mkldnn_param.min_calib_range.value();
        *max_output_ptr = mkldnn_param.max_calib_range.value();

        full_param_.requantize_scales[0] = quantized_out_range /
          MaxAbs(*min_output_ptr, *max_output_ptr) / data_scale / weight_scale;
      } else {
        Stream<cpu> *s = ctx.get_stream<cpu>();
        mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
          min_output_ptr, max_output_ptr, &min_data, &max_data, &min_weight, &max_weight);
      }
    }

    fwd_.reset(new MKLDNNFullyConnectedForward(full_param_, ctx.is_train, data, weight,
      (has_bias ? &cached_bias_ : nullptr), out_md));
    initialized_ = true;
  }
  std::vector<NDArray> new_inputs;
  std::vector<OpReqType> new_req;
  if (has_bias) {
    new_inputs = {data, weight, cached_bias_};
    new_req = {req[fullc::kData], req[fullc::kWeight], req[fullc::kBias]};
  } else {
    new_inputs = {data, weight};
    new_req = {req[fullc::kData], req[fullc::kWeight]};
  }

  MKLDNNFCForwardFullFeature(full_param_, ctx, fwd_.get(), new_inputs, new_req, out_data);
}

static void SgMKLDNNFCParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNFCFullParam full_param;
  try {
    full_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "FullyConnected") {
      full_param.default_param =
          nnvm::get<FullyConnectedParam>(node->attrs.parsed);
    }
  });
  attrs->parsed = std::move(full_param);
}

static std::vector<std::string> SgMKLDNNFCListInputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (full_param.mkldnn_param.quantized) {
    input_names.emplace_back("min_data");
    input_names.emplace_back("max_data");
    input_names.emplace_back("min_weight");
    input_names.emplace_back("max_weight");
    if (!full_param.default_param.no_bias) {
      input_names.emplace_back("min_bias");
      input_names.emplace_back("max_bias");
    }
  }
  return input_names;
}

static std::vector<std::string> SgMKLDNNFCListOutputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    if (full_param.mkldnn_param.enable_float_output)
      return std::vector<std::string>{"output"};
    else
      return std::vector<std::string>{"output", "min_output", "max_output"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

template <typename T>
static inline void FillBaseInputOutputInfo(const FullyConnectedParam &param,
                                           std::vector<T> *base_in_attrs,
                                           std::vector<T> *base_out_attrs,
                                           std::vector<T> *in_attrs,
                                           std::vector<T> *out_attrs) {
  auto base_num_inputs = param.no_bias ? 2 : 3;

  base_out_attrs->push_back(out_attrs->at(0));
  for (int i = 0; i < base_num_inputs; ++i) {
    base_in_attrs->push_back(in_attrs->at(i));
  }
}

static bool SgMKLDNNFCInferShape(const nnvm::NodeAttrs &attrs,
                                 mxnet::ShapeVector *in_shapes,
                                 mxnet::ShapeVector *out_shapes) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;
    FillBaseInputOutputInfo(full_param.default_param, &base_in_shapes, &base_out_shapes,
                            in_shapes, out_shapes);
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);

    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }

    out_shapes->at(0) = base_out_shapes[0];
    if (!full_param.mkldnn_param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNFCInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    size_t base_num_inputs = full_param.default_param.no_bias ? 2 : 3;

    // TODO(ciyong): currently, only uint8 fully_connected is upported,
    // int8 fully_connected will be supported after mkldnn v0.18
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kUint8);
    for (size_t i = 1; i < in_types->size(); ++i) {
      if (i < base_num_inputs) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      }
    }

    if (full_param.mkldnn_param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else {
      if (full_param.mkldnn_param.min_calib_range.has_value() &&
          full_param.mkldnn_param.max_calib_range.has_value()) {
        if (full_param.mkldnn_param.with_relu) {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
        } else {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
        }
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNFCStorageType(const nnvm::NodeAttrs &attrs,
                                  const int dev_mask,
                                  DispatchMode *dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs;
    FillBaseInputOutputInfo(full_param.default_param, &base_in_attrs, &base_out_attrs,
                            in_attrs, out_attrs);
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                            &base_in_attrs, &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }

    out_attrs->at(0) = base_out_attrs[0];
    if (!full_param.mkldnn_param.enable_float_output) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_attrs, out_attrs);
  }
}

static OpStatePtr CreateSgMKLDNNFCState(const nnvm::NodeAttrs &attrs,
                                        Context ctx,
                                        const mxnet::ShapeVector &in_shapes,
                                        const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNFCOp>(attrs);
}

static void SgMKLDNNFCForward(const OpStatePtr &state_pointer,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  SgMKLDNNFCOp &op = state_pointer.get_state<SgMKLDNNFCOp>();
  op.Forward(ctx, inputs, req, outputs);
}

nnvm::NodePtr SgMKLDNNFCQuantizedOp(const NodeAttrs& attrs) {
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_fully_connected");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

NNVM_REGISTER_OP(_sg_mkldnn_fully_connected)
.describe(R"code(_sg_mkldnn_fully_connected)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  auto num_inputs = full_param.default_param.no_bias ? 2 : 3;
  if (full_param.mkldnn_param.quantized)
    return num_inputs * 3;
  else
    return num_inputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  return (full_param.mkldnn_param.quantized &&
          !full_param.mkldnn_param.enable_float_output) ? 3 : 1;
})
.set_attr_parser(SgMKLDNNFCParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames", SgMKLDNNFCListInputNames)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", SgMKLDNNFCListOutputNames)
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNFCInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNFCInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNFCStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNFCState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNFCForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                               DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNFCQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; });

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
