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

#ifndef MXNET_NGRAPH_NGRAPH_EMITTER_H_
#define MXNET_NGRAPH_NGRAPH_EMITTER_H_

#include <map>
#include <string>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {

// Alias for maps of name, function, where function returns an ngraph node
using OpEmitter =
    std::map<std::string, std::function<NgraphNodePtr(const NodePtr&)> >;

// Emitter primairily serves to create and store ngraph Nodes
class Emitter {
 public:
  Emitter();
  // maps of ngraph operation generator functions
  OpEmitter ngraph_op_funcs_;
  // maps of ngraph operations specialized for training
  OpEmitter ngraph_op_funcs_train_;

 protected:
  // create unary operation functions
  void CreateUnaryOps();
  // create binary operation functions
  void CreateBinaryOps();
  // create larger MXNet layer operations
  void CreateLayerOps();
  // Factory function for autobroadcasting the inputs of a node
  template <class op>
  NgraphNodePtr CreateAutoBroadcast(const NodePtr& node);
  // Factory function for reducing based on a reduction op function
  NgraphNodePtr ReduceAxes(
      const NgraphNodePtr& node, ngraph::AxisVector axes, bool exclude,
      bool keepdims,
      const std::function<NgraphNodePtr(const NgraphNodePtr&,
                                        const ngraph::AxisSet&)>& func);
  NgraphNodePtr ReduceAxes(
      const NodePtr& node,
      const std::function<NgraphNodePtr(const NgraphNodePtr&,
                                        const ngraph::AxisSet&)>& func);

  NgraphNodePtr BatchNorm(const NodePtr& node, const bool is_train);

  // information on compiled objects
  std::map<NodePtr, NgraphNodePtr> op_map_;
  // ops require special handling for training
  std::map<NodePtr, NgraphNodePtr> op_map_train_;
  std::vector<NodePtr> placeholder_order_;
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_EMITTER_H_
