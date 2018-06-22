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

#include "ops/slice.h"
#include "ngraph_sgcompiler_utils.h"
#include "../operator/tensor/matrix_op-inl.h"

namespace ngraph_bridge {

NgraphNodePtr create_slice_op(
    const NgraphNodePtr& node,
    const nnvm::NodeAttrs& attrs) {

    const mxnet::op::SliceParam& param = nnvm::get<mxnet::op::SliceParam>(attrs.parsed);
    nnvm::TShape tshape = NShape_to_TShape(node->get_shape());
    ngraph::Coordinate ng_begin, ng_end, ng_step;
    ngraph::AxisSet axes;
    bool reverse = false;

    for (mxnet::index_t i = 0; i < param.begin.ndim(); ++i) {
        int b = 0, e = tshape[i], s = 1;
        const int len = tshape[i];

        if (param.step.ndim() != 0U) {
            const auto& opt_step_val = param.step[i];
            if (opt_step_val.has_value()) {
                s = opt_step_val.value();
            }
        }

        if (param.begin[i].has_value()) {
            b = param.begin[i].value();
            if (b < 0) {
                b = abs(b);
            }
        } else if (s < 0) {
            b = 0;
        }

        if (param.end[i].has_value()) {
            e = param.end[i].value();
            if (e < 0) {
                e = abs(e);
            }
        } else if (s < 0) {
            e = len;
        }

        if (s < 0) {
            reverse = true;
            s = abs(s);
            axes.insert(i);
            if (b > e) {
                size_t temp = b;
                b = e;
                e = temp;
            }
        }

        ng_begin.push_back(b);
        ng_end.push_back(e);
        ng_step.push_back(s);
    }

    for (mxnet::index_t i = param.begin.ndim(); i < tshape.ndim(); ++i) {
        ng_begin.push_back(0);
        ng_end.push_back(tshape[i]);
        ng_step.push_back(1);
    }

    NgraphNodePtr slice;
    if (reverse) {
        slice = std::make_shared<ngraph::op::Slice>(
                std::make_shared<ngraph::op::Reverse>(
                    node, ngraph::AxisSet{axes}),
                ng_begin, ng_end, ng_step);
    } else {
        slice = std::make_shared<ngraph::op::Slice>(node, ng_begin, ng_end, ng_step);
    }
    return slice;
}
}  // namespace ngraph_bridge
