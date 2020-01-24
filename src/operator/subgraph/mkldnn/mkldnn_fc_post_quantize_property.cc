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
 * \file mkldnn_fc_post_quantize_property.cc
 * \brief Partition gragph property for MKLDNN Quantized FullyConnected operator
 * \author Ciyong Chen
*/

#if MXNET_USE_MKLDNN == 1

#include "../common.h"
#include "../subgraph_property.h"
#include "../../nn/fully_connected-inl.h"
#include "../../quantization/requantize-inl.h"

namespace mxnet {
namespace op {

#define QUANTIZED_FC_NAME "_sg_mkldnn_fully_connected"

class SgMKLDNNFCPostQuantizeSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kRequantize,
    kSuccess,
  };

 private:
  bool disable_all;
  bool disable_float_output;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  explicit SgMKLDNNFCPostQuantizeSelector(const bool dis_all,
                                          const bool dis_float_output)
      : disable_all(dis_all),
        disable_float_output(dis_float_output) {}

  bool Select(const nnvm::Node &n) override {
    if ((!disable_all) && n.op() == Op::Get(QUANTIZED_FC_NAME)) {
      status = disable_all ? kSuccess : kStart;
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (status == kFail || status == kSuccess || new_node.is_variable())
      return false;
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list.back() != &n) {
      if (std::find(matched_list.begin(), matched_list.end(), &n) !=
        matched_list.end()) {
        while (matched_list.back() != &n) {
          matched_list.pop_back();
        }
      }

      status = kSuccess;
      return false;
    }

    switch (status) {
      case kStart:
        if (new_node.op() == Op::Get("_contrib_requantize")) {
          auto const &param = nnvm::get<RequantizeParam>(new_node.attrs.parsed);
          if (param.min_calib_range.has_value() &&
              param.max_calib_range.has_value()) {
            matched_list.push_back(&new_node);
            status = kRequantize;
            return true;
          }
        }
      case kRequantize:
        if ((!disable_float_output) && (new_node.op() == Op::Get("_contrib_dequantize"))) {
            matched_list.push_back(&new_node);
            status = kSuccess;
            return true;
        }
      default:
        status = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if ((status != kSuccess) || (matched_list.size() <= 1)) {
      return std::vector<nnvm::Node *>(0);
    } else {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_list) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }
};

class SgMKLDNNFCPostQuantizeProperty : public SubgraphProperty {
 public:
  SgMKLDNNFCPostQuantizeProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_POST_OPT", false);
    disable_fuse_all = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_QFC_FUSE_ALL", false);
    disable_float_output = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_QFC_FLOAT_OUTPUT", false);

    disable_all = disable_all || disable_fuse_all;
    if (disable_all) {
      LOG(INFO) << "MKLDNN FullyConnected post-quantization optimization pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN FullyConected post-quantization optimization pass.";
    }
  }

  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNFCPostQuantizeProperty>();
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr fc_node = nullptr;
    nnvm::NodePtr requantize_node = nullptr;
    nnvm::NodePtr dequantize_node = nullptr;

    DFSVisit(sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      if (node->op() == Op::Get(QUANTIZED_FC_NAME)) {
        fc_node = node;
      } else if (node->op() == Op::Get("_contrib_requantize")) {
        requantize_node = node;
      } else if (node->op() == Op::Get("_contrib_dequantize")) {
        dequantize_node = node;
      }
    });

    CHECK_NOTNULL(fc_node);
    CHECK_NOTNULL(requantize_node);
    auto const &requantize_param =
        nnvm::get<RequantizeParam>(requantize_node->attrs.parsed);
    CHECK(requantize_param.min_calib_range.has_value());
    CHECK(requantize_param.max_calib_range.has_value());

    // When only fused quantized_fullyconnected and requantize, set min/max_cablib_range,
    // When fused quantized_fullyconnected + requantize + dequantize, set dequantize flag to true.
    if (dequantize_node != nullptr) {
      fc_node->attrs.dict["enable_float_output"] = "True";
    } else {
      fc_node->attrs.dict["min_calib_range"] =
          std::to_string(requantize_param.min_calib_range.value());
      fc_node->attrs.dict["max_calib_range"] =
          std::to_string(requantize_param.max_calib_range.value());
    }
    fc_node->op()->attr_parser(&(fc_node->attrs));
    return fc_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector =
        std::make_shared<SgMKLDNNFCPostQuantizeSelector>(disable_all,
                                                         disable_float_output);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  bool disable_all;
  bool disable_fuse_all;
  bool disable_float_output;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_POST_FC_QUANTIZE, SgMKLDNNFCPostQuantizeProperty);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
