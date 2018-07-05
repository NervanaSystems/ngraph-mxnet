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
 * \file mkldnn_softmax.cc
 * \brief
 * \author Da Zheng
*/

#include "../softmax-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNCopy(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                const NDArray &in_data, const OpReqType &req,
                const NDArray &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);

  // If the input data is a view of an MKLDNN array, we should create a new
  // NDArray with reordered data.
  NDArray data = in_data;
  if (data.IsMKLDNNData() && data.IsView())
    data = data.Reorder2Default();
  auto in_mem = data.GetMKLDNNData();
  if (req == kAddTo) {
    TmpMemMgr::Get()->Init(ctx.requested[0]);
    // We should try and force the output memory has the same format
    // as the input memory. If not, we'll have to reorder memory.
    auto out_mem = out_data.GetMKLDNNData(in_mem->get_primitive_desc());
    if (out_mem == nullptr)
      out_mem = out_data.GetMKLDNNData();
    auto sum_res = TmpMemMgr::Get()->Alloc(out_mem->get_primitive_desc());
    MKLDNNSum(*in_mem, *out_mem, *sum_res);
    const_cast<NDArray &>(out_data).CopyFrom(*sum_res);
  } else {
    const_cast<NDArray &>(out_data).CopyFrom(*in_mem);
  }
  MKLDNNStream::Get()->Submit();
}

}   // namespace op
}   // namespace mxnet
#endif
