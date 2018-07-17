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

#ifndef MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include "../imperative/cached_op.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

/*
 * Infer the data types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphDataType(const nnvm::Symbol &subgraph, std::vector<int> *in_type,
                           std::vector<int> *out_type);

/*
 * Infer the shape of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphShape(const nnvm::Symbol &subgraph,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape);

/*
 * Infer the storage types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs);

/*
 * This contains the states for running a loop and provides methods
 * of running the subgraph computation for an iteration.
 */
class LoopState {
  // These are output arrays from all iterations.
  // They also contain the Op state for each CachedOp.
  std::vector<std::vector<NDArray> > all_outputs;
  std::vector<std::vector<NDArray> > all_inputs;
  // For inference, there should be only one cached op because we
  // want to share the memory in iterations.
  // For training, each iteration has a cached op because each iteration
  // needs to maintain a set of memory buffers for all computation states,
  // which will be used in the backward.
  CachedOpPtr iter_op;
  std::vector<OpStatePtr> all_states;
  Symbol subgraph_sym;
  nnvm::Graph subgraph;

 public:
  explicit LoopState(const Symbol &g);

  void Forward(int iter_no,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray> &outputs,
               bool is_recording);
  void Backward(int iter_no,
                const std::vector<NDArray> &ograds,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &igrads);
  void Cleanup() {
    all_outputs.clear();
    all_inputs.clear();
    all_states.clear();
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
