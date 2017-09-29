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

#ifndef NGRAPH_PYCOMPILER_H_
#define NGRAPH_PYCOMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_utils.h"
#include "ngraph_pyemitter.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using axes_pair = std::pair<std::string, int>;
using axes_map = std::map<axes_pair, py::object>;
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

class PyCompiler : public PyEmitter {
 public:
  PyCompiler(){};
  std::shared_ptr<Graph> Compile(NodePtr graph);
 private:
  // Generator to create functions that convert mxnet layer operations
  // into a series of ngraph operations
  layerGraphs create_layerGraphs();
  // parse the nnvm graph into an intermediate rep
  Graph ParseNNVMGraph(nnvm::Graph& graph);
  // check nodes against ngraph operations
  void CheckInNGraph(Graph& graph);
  // compile subgraph into ngraph python objects
  void CompileSubgraph(std::shared_ptr<Graph> graph);
  // compile inputs to a node
  void CompileInput(NodePtr input, axes_map node_axes);
  void CompileInputs(NodePtr node);
  // compile a single node into an ngraph python object
  void CompileNode(NodePtr node, std::shared_ptr<Graph> graph);
  void ClearOpMap();
};

}  // end namespace ngraph
#endif