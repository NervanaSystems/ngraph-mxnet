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

#ifndef NGRAPH_EMITTER_H_
#define NGRAPH_EMITTER_H_

#include "ngraph_graph.h"
#include "ngraph_autobroadcast.h"

using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

namespace ngraph_bridge {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject

using OpEmitter =
    std::map<std::string,
             std::function<NgraphNodePtr(const NodePtr&)> >;

class Emitter {
public:
  Emitter();
  // maps of ngraph operation generator functions
  OpEmitter ngraph_op_funcs_;
protected:
  // create unary operation functions
  void CreateUnaryOps();
  // create binary operation functions
  void CreateBinaryOps();
  // create larger MXNet layer operations
  void CreateLayerOps();

  AutoBroadcast CreateAutoBroadcast(const NodePtr& node);

  // information on compiled objects
  std::map<NodePtr, NgraphNodePtr> op_map_;
  std::vector<NodePtr> placeholder_order_;
};

}  // end namespace ngraph
#endif
