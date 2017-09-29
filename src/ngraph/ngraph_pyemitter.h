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

#ifndef NGRAPH_PYEMITTER_H_
#define NGRAPH_PYEMITTER_H_

#include "ngraph_utils.h"
#include "ngraph_graph.h"

namespace ngraph {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject
using axes_pair = std::pair<std::string, int>;
using axes_map = std::map<axes_pair, py::object>;
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

using UnaryOps =
    std::map<std::string,
             std::function<py::object(const py::object&, const std::string&)> >;
using BinaryOps =
    std::map<std::string,
             std::function<py::object(const py::object&, const py::object&,
                                      const std::string&)> >;
using LayerOps =
    std::map<std::string,
             std::function<py::object(const NodePtr&, py::object)> >;

class PyEmitter {
public:
  PyEmitter();
  // vector of available operations
  std::vector<std::string> NgraphOps_;
protected:
  // create unary operation functions
  UnaryOps create_UnaryOps(const py::module& ng);
  // create binary operation functions
  BinaryOps create_BinaryOps(const py::module& ng);
  // create larger MXNet layer operations
  LayerOps create_LayerOps(const py::module& ng);

  // create variable objects in ngraph
  py::object createPyPlaceholder(std::string name, py::tuple axes);
  py::object make_axis(int length, std::string name);
  py::object match_axes(const py::object& lhs, const py::object& rhs);

  // maps of ngraph operation generator functions
  UnaryOps NgraphUnaryOps_;
  BinaryOps NgraphBinaryOps_;
  LayerOps NgraphLayerOps_;

  // python modules and objects required for compilation/computation
  py::module ng_;
  py::module ngt_;
  py::object transformer_;

  // information on compiled objects
  std::map<std::string, py::object> op_map;
  std::vector<std::string> placeholder_order;
};

}  // end namespace ngraph
#endif