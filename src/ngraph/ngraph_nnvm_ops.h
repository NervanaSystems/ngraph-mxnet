// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_

#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>

#include <string>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {
// function for returning nnvm::Op corresponding to a subgraph
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph);
// function for registering subgraph operation with nnvm
void register_subgraph(std::shared_ptr<Graph> graph);
// function for computing forward on ngraph
void compute_forward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                     const std::vector<mxnet::TBlob>& inputs,
                     const std::vector<mxnet::TBlob>& outputs);
// function for computing backward on ngraph
void compute_backward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                      const std::vector<mxnet::TBlob>& inputs,
                      const std::vector<mxnet::TBlob>& outputs);

// dummy parameter struct to match mxnet API
struct NGraphParam {
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  void Init(const nnvm::NodeAttrs& attrs) {}
  // Clean up the graph when this param is deleted
  // if we have 3 or fewer references left
  // forward func/backward func/this param object
  ~NGraphParam() {
    if (g != nullptr && g.use_count() <= 3) {
      g->CleanUp();
    }
  }
  std::shared_ptr<ngraph_bridge::Graph> g;
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
