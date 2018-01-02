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

#include <nnvm/op.h>

#include <string>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {
// function for returning nnvm::Op corresponding to a subgraph
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph);
// function for registering subgraph operation with nnvm
void register_subgraph(std::shared_ptr<Graph> graph);

// dummy parameter struct to match mxnet API
struct NGraphParam {
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  void Init(const nnvm::NodeAttrs& attrs) {}
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
