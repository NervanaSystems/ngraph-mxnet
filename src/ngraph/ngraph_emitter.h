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
  OpEmitter NgraphOpFuncs_;
protected:
  // create unary operation functions
  void create_UnaryOps();
  // create binary operation functions
  void create_BinaryOps();
  // create larger MXNet layer operations
  void create_LayerOps();

  // information on compiled objects
  std::map<NodePtr, NgraphNodePtr> op_map;
  std::vector<NodePtr> placeholder_order;
};

}  // end namespace ngraph
#endif