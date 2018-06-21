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

#include <vector>
#include "ops/slice.h"
#include "ngraph_sgcompiler_utils.h"
#include "../operator/tensor/matrix_op-inl.h"

namespace ngraph_bridge {

NgraphNodePtr create_slice_op(
    const NgraphNodePtr& node,
    const nnvm::NodeAttrs& attrs) {

    const mxnet::op::SliceParam& param = nnvm::get<mxnet::op::SliceParam>(attrs.parsed);
    ngraph::Shape nshape = node->get_shape();
    nnvm::TShape tshape = NShape_to_TShape(nshape);
    ngraph::Coordinate ng_begin, ng_end, ng_step;
    std::vector<int> begin_val, end_val, step_val;

    MXNET_NDIM_SWITCH(tshape.ndim(), ndim, {
      mxnet::common::StaticArray<int, ndim> begin, end, step;
      mxnet::op::GetIndexRange(tshape, param.begin, param.end, param.step, &begin, &end, &step);
      for (mxnet::index_t i = 0; i < param.begin.ndim(); ++i) {
        const int b = begin[i], e = end[i], s = step[i];
        begin_val.push_back(b);
        end_val.push_back(e);
        step_val.push_back(s);
      }
    });

    for (size_t i = 0; i < begin_val.size(); ++i) {
      ng_begin.push_back(begin_val[i] < 0 ? nshape.size() + begin_val[i] : begin_val[i]);
      ng_end.push_back(end_val[i] < 0 ? nshape.size() + end_val[i] : end_val[i]);
      ng_step.push_back(step_val[i] < 0 ? nshape.size() + step_val[i] : step_val[i]);
    }

    NgraphNodePtr slice = std::make_shared<ngraph::op::Slice>(node, ng_begin, ng_end, ng_step);
    return slice;
}
}  // namespace ngraph_bridge
