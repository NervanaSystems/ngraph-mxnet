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
  void setExeMode(GraphExeMode exe_mode);

 protected:
  void ClearOpMap();
  template <class op>
  NgraphNodePtr CreateAutoBroadcast(const NodePtr& node);
  template <class op>
  NgraphNodePtr CreateScalarOp(const NodePtr& node);
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

  // information on compiled objects
  std::map<NodePtr, NgraphNodePtr> op_map_;
  std::map<NodePtr, NgraphNodePtr> aux_op_map_;
  std::vector<NodePtr> placeholder_order_;
  GraphExeMode exe_mode_;

 private:
  void InitOpFuncs();
  // create unary operation functions
  void CreateUnaryOps();
  // create binary operation functions
  void CreateBinaryOps();
  // create larger MXNet layer operations
  void CreateLayerOps();
  // Factory function for autobroadcasting the inputs of a node

 protected:
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_EMITTER_H_
