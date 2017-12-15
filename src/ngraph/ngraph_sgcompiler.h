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

#ifndef NGRAPH_SGCOMPILER_H_
#define NGRAPH_SGCOMPILER_H_

#include "ngraph_emitter.h"
#include "ngraph_graph.h"

namespace ngraph_bridge {

class SGCompiler : public Emitter {
 public:
  std::shared_ptr<Graph> Compile(NodePtr sub_graph);

 protected:
  // compile subgraph into ngraph objects
  void CompileSubgraph(std::shared_ptr<Graph> sub_graph);
  // compile input to a node
  void CompileInput(NodePtr input);
  // compile the graph nodes into ngraph objects
  void CompileNodes(NodePtr node, const std::shared_ptr<Graph> sub_graph);
  void ClearOpMap();
};

}  // namespace ngraph_bridge
#endif
