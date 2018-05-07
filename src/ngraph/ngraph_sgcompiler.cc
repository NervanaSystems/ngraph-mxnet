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

#include "ngraph_sgcompiler.h"

#include <nnvm/node.h>
#include <nnvm/pass.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/reshape_elimination.hpp>
#include <ngraph/runtime/cpu/pass/cpu_fusion.hpp>
#include <ngraph/serializer.hpp>

#include "ngraph_sgcompiler_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

void CompileForward(std::shared_ptr<Graph> sub_graph,
                    std::shared_ptr<ngraph::Function> f,
                    GraphExeMode exe_mode) {
  const int mode = static_cast<int>(exe_mode);

  auto backend = GetBackendFromContext(sub_graph->context_);

  // Log the graph so Graph_* corresponds to Function_* in codgen
  if (ngraph_log_graph()) {
    dump_graph(f, __func__, "fprop");
  }
  auto results = f->get_results();
  for (size_t i = 0; i < sub_graph->num_outputs_; ++i)
    results[i]->set_needs_default_layout(true);

  backend->compile(f);
  sub_graph->ngraph_forward[mode] = f;
}

void CompileForwardBackward(std::shared_ptr<Graph> sub_graph,
                            std::shared_ptr<ngraph::Function> f,
                            std::shared_ptr<ngraph::Function> bf,
                            GraphExeMode exe_mode,
                            const ngraph::FpropCache &fprop_cache) {
  const int mode = static_cast<int>(exe_mode);

  auto backend = GetBackendFromContext(sub_graph->context_);

  // clone the functions to ensure we don't have
  // any repeated nodes between graphs
  ngraph::NodeMap fmap;
  ngraph::NodeMap bfmap;

  auto f_copy = ngraph::clone_function(*f, fmap);
  auto bf_copy = ngraph::clone_function(*bf, bfmap);

  // Log the graphs so Graph_* corresponds to Function_* in codgen
  if (ngraph_log_graph()) {
    dump_graph(f_copy, __func__, "fprop");
    dump_graph(bf_copy, __func__, "bprop");
  }

  auto results = f_copy->get_results();
  for (size_t i = 0; i < (sub_graph->num_outputs_ +
                          sub_graph->cached_aux_values[mode].size());
       ++i)
    results[i]->set_needs_default_layout(true);

  backend->compile(f_copy);

  for (auto result : f->get_results()) {
    if (fprop_cache.node_param_map->exists(result->get_argument(0))) {
      auto cloned_result = fmap.get(result);
      auto bf_param = fprop_cache.node_param_map->get(result->get_argument(0));
      auto cloned_bf_param = bfmap.get(bf_param);
      auto layout =
          cloned_result->get_output_tensor_view()->get_tensor_view_layout();
      cloned_bf_param->get_output_tensor_view()->set_tensor_view_layout(layout);
    }
  }

  for (auto res : bf_copy->get_results()) res->set_needs_default_layout(true);
  backend->compile(bf_copy);

  sub_graph->ngraph_forward[mode] = f_copy;
  sub_graph->ngraph_backward[mode] = bf_copy;
}

void OptimizeGraph(std::shared_ptr<Graph> sub_graph,
                   std::shared_ptr<ngraph::Function> f,
                   std::shared_ptr<ngraph::Function> bf) {
  // start by removing excess reshapes
  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
  pass_manager.register_pass<ngraph::pass::ReshapeElimination>();

  pass_manager.run_passes(f);
  pass_manager.run_passes(bf);

  if (sub_graph->context_ == mxnet::Context::CPU()) {
    // if we're in CPU, combine the graphs
    ngraph::NodeVector dYdXs;
    for (size_t i = 0; i < bf->get_output_size(); ++i) {
      dYdXs.push_back(bf->get_output_op(i)->get_argument(0));
    }
    ngraph::NodeVector combined_outputs{f->get_output_op(0)->get_argument(0)};
    combined_outputs.insert(combined_outputs.end(), dYdXs.begin(), dYdXs.end());

    std::vector<std::shared_ptr<ngraph::op::Parameter>> combined_parameters =
        f->get_parameters();
    std::vector<std::shared_ptr<ngraph::op::Parameter>> back_parameters =
        bf->get_parameters();

    combined_parameters.insert(combined_parameters.end(),
                               back_parameters.begin(), back_parameters.end());
    auto combinedf = std::make_shared<ngraph::Function>(combined_outputs,
                                                        combined_parameters);
    // rerun Reshape elimination to help simplify the graph again, run CPUFusion
    // this replaces nodes in both f and bf due to shared-ptr - ness
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<ngraph::runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(combinedf);
  }
}

// Main compilation function
std::shared_ptr<Graph> SGCompiler::Compile(NodePtr sub_graph) {
  // clear the op_map_ and placeholder_order
  ClearOpMap();
  // cast the graph
  auto sg = std::dynamic_pointer_cast<Graph>(sub_graph);

  CompileSubgraph(sg);

  return sg;
}

std::shared_ptr<ngraph::Function> SGCompiler::MakeForwardFunction(
    std::shared_ptr<Graph> sub_graph) {
  ngraph::op::ParameterVector parameters;

  for (const auto input : placeholder_order_) {
    // get the parameters
    parameters.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(op_map_.at(input)));
  }

  const int mode = static_cast<int>(exe_mode_);

  // default output
  ngraph::NodeVector outputs;
  for (auto output : sub_graph->outputs_) {
    outputs.push_back(op_map_.at(output));
  }

  auto backend = GetBackendFromContext(sub_graph->context_);

  // push additional aux outputs
  if (exe_mode_ == GraphExeMode::kTrain && !aux_op_map_.empty()) {
    int i = 0;
    for (auto input : sub_graph->inputs_) {
      if (aux_op_map_.count(input)) {
        NgraphNodePtr ngraph_node = aux_op_map_.at(input);
        outputs.push_back(ngraph_node);

        // cache aux node
        sub_graph->cached_aux_values[mode].push_back(backend->create_tensor(
            ngraph_node->get_element_type(), ngraph_node->get_shape()));
        sub_graph->cached_aux_positions[mode].push_back(i);
      }
      i += 1;
    }
  }

  // create the Forward Function object representing the graph
  return std::make_shared<ngraph::Function>(outputs, parameters);
}

std::pair<std::shared_ptr<ngraph::Function>,
          std::vector<std::shared_ptr<ngraph::Node>>>
SGCompiler::MakeBackwardFunction(std::shared_ptr<Graph> sub_graph,
                                 std::shared_ptr<ngraph::Function> f) {
  // get parameters
  std::vector<std::shared_ptr<ngraph::op::Parameter>> back_parameters =
      f->get_parameters();

  ngraph::NodeVector adjoints;
  ngraph::NodeVector outputs;
  for (auto node : sub_graph->outputs_) {
    // Get the output
    auto Y = op_map_.at(node);
    // Create the Adjoint
    NgraphNodePtr C = std::make_shared<ngraph::op::Parameter>(
        Y->get_element_type(), Y->get_shape());
    outputs.push_back(Y);
    adjoints.push_back(C);
  }

  ngraph::autodiff::Adjoints adjoint{outputs, adjoints};

  // Perform autodiff
  std::vector<NgraphNodePtr> dYdXs(back_parameters.size());
  transform(
      back_parameters.begin(), back_parameters.end(), dYdXs.begin(),
      [&adjoint](const NgraphNodePtr &X) { return adjoint.backprop_node(X); });

  // create the backward function
  std::vector<std::shared_ptr<ngraph::op::Parameter>> param_adjoints;
  for (auto n : adjoints)
    param_adjoints.push_back(
        std::dynamic_pointer_cast<ngraph::op::Parameter>(n));
  back_parameters.insert(back_parameters.begin(), param_adjoints.begin(),
                         param_adjoints.end());

  return {std::make_shared<ngraph::Function>(dYdXs, back_parameters), adjoints};
}

// Compile a Subgraph into ngraph forward and backward call frames
void SGCompiler::CompileSubgraph(std::shared_ptr<Graph> sub_graph) {
  auto backend = GetBackendFromContext(sub_graph->context_);

  // initalize a placeholder order vector for this subgraph
  for (auto i : sub_graph->inputs_) placeholder_order_.push_back(i);

  // compile all the nodes in the graph
  for (auto output : sub_graph->outputs_) {
    CompileNodes(output, sub_graph);
  }

  auto f = MakeForwardFunction(sub_graph);
  if (ngraph_log_graph()) {
    dump_graph(f, __func__, "pre-optimized-fprop");
  }

  std::shared_ptr<ngraph::Function> maybe_bf;
  std::vector<std::shared_ptr<ngraph::Node>> adjoints;
  if (exe_mode_ == GraphExeMode::kTrain) {
    auto bfa = MakeBackwardFunction(sub_graph, f);
    maybe_bf = bfa.first;
    adjoints = bfa.second;
    if (ngraph_log_graph()) {
      dump_graph(maybe_bf, __func__, "pre-optimized-bprop");
    }

    // OptimizeGraph's real benefit comes from optimizing the fprop cache, so we
    // only call it when
    // we're in training mode...
    OptimizeGraph(sub_graph, f, maybe_bf);
  }

  if (ngraph_log_graph()) {
    dump_graph(f, __func__, "post-optimized-fprop");

    if (maybe_bf) {
      dump_graph(maybe_bf, __func__, "post-optimized-bprop");
    }
  }

  if (sub_graph->enable_fprop_cache && exe_mode_ == GraphExeMode::kTrain) {
    auto fprop_cache = ngraph::cache_fprop(f, maybe_bf, adjoints);

    if (ngraph_log_graph()) {
      dump_graph(fprop_cache.fprop, __func__, "fprop_cache.fprop");
      dump_graph(fprop_cache.bprop, __func__, "fprop_cache.bprop");
    }

    CompileForwardBackward(sub_graph, fprop_cache.fprop, fprop_cache.bprop,
                           exe_mode_, fprop_cache);

    for (auto node : fprop_cache.fprop_output_nodes) {
      sub_graph->cached_values[static_cast<int>(exe_mode_)].push_back(
          backend->create_tensor(node->get_element_type(), node->get_shape()));
    }

    return;
  }

  if (exe_mode_ == GraphExeMode::kTrain) {
    ngraph::FpropCache fprop_cache;
    fprop_cache.node_param_map = std::make_shared<ngraph::NodeMap>();
    CompileForwardBackward(sub_graph, f, maybe_bf, exe_mode_, fprop_cache);
    return;
  }

  CHECK(exe_mode_ == GraphExeMode::kInfer);
  // No need to compile the backprop function if we're running in inference
  // mode.
  CompileForward(sub_graph, f, exe_mode_);
}

/**
 * Function to perform a graph pass and compile all of the nodes
 * need to make sure we compile the inputs of a node before the node itself
 **/
void SGCompiler::CompileNodes(NodePtr node,
                              const std::shared_ptr<Graph> sub_graph) {
  GraphVisitor visitor;

  // the operation of this graph traverse compiles the node as
  // an input if it's not part of the subgraph or as an ngraph operation
  // if the node is part of the subgraph
  // we capture this so we can save the outputs to the SGCompiler op_map_
  visitor.operation = [this, sub_graph](NodePtr node) {
    if (!op_map_.count(node)) {
      // if it's not in the graph, it's an input, compile it as an input
      if (!in_vec(sub_graph->nodes_, node)) {
        this->CompileInput(node);
      } else {
        this->op_map_.insert(
            {node, this->ngraph_op_funcs_[node->operation_](node)});

        // Verify that the shapes computed by NNVM and nGraph are identical...
        const nnvm::TShape &nnvm_shape = node->shape_;
        const std::shared_ptr<ngraph::Node> ngraph_node = this->op_map_[node];
        const ngraph::Shape &ngraph_provided_shape = ngraph_node->get_shape();
        const nnvm::TShape ngraph_shape_as_nnvm_shape =
            NShape_to_TShape(ngraph_provided_shape);

        bool bad_shape = false;
        // nGraph represent scalars as 0-dim tensors
        // nnvm::TShape represents them as 1-dim tensors of size 1
        // if nGraph is returing a 0-dim tensor, the nnvm
        // shape should be size 1
        if (((ngraph_provided_shape.size() == 0) &&
             !(nnvm_shape.ndim() == 1 && nnvm_shape[0] == 1))) {
          bad_shape = true;
          // if ngraph is returning a finitely-sized tensor,
          // the two shapes should match
        } else if ((ngraph_provided_shape.size() != 0) &&
                   (nnvm_shape != ngraph_shape_as_nnvm_shape)) {
          bad_shape = true;
        }
        // if either of those conditions doesn't hold, throw an error
        if (bad_shape) {
          std::ostringstream os;
          os << "NGRAPH_BRIDGE: In " << __PRETTY_FUNCTION__ << " : "
             << std::endl;
          os << "   Error processing node: " << node->createNodeLabel()
             << std::endl;
          os << "   Shape mismatch:"
             << " nnvm::Tshape=" << nnvm_shape
             << ", ngraph::Shape=" << ngraph_shape_as_nnvm_shape;
          throw std::runtime_error(os.str());
        }

        // Verify that the element-types computed by NNVM and nGraph are
        // identical...
        const ngraph::element::Type &ng_type = ngraph_node->get_element_type();
        const ngraph::element::Type &nnvm_type_as_ng_type =
            getType(node->dtype_);
        if (ng_type != nnvm_type_as_ng_type) {
          std::ostringstream os;
          os << "NGRAPH_BRIDGE: In " << __PRETTY_FUNCTION__ << " : "
             << std::endl;
          os << "   Error processing node: " << node->createNodeLabel()
             << std::endl;
          os << "   element-type mismatch: NNVM elem-type=" << node->dtype_
             << ", nGraph node's elem-type=" << ng_type;
          throw std::runtime_error(os.str());
        }
      }
    }
  };

  visitor.stop_condition = [this, sub_graph](NodePtr node, NodePtr input) {
    // continue if...
    // 1) node is in subgraph or a subgraph input
    // 2) input not visited
    if (in_vec(sub_graph->nodes_, node) && !(this->op_map_.count(input))) {
      return false;
    }
    // else, stop traversing the graph
    return true;
  };

  GraphTraverse(node, visitor);
}

// Compile the inputs
void SGCompiler::CompileInput(NodePtr input) {
  auto shape = TShape_to_NShape(input->shape_);
  // make a shaped and typed parameter based on the input node
  // store it in the op_map_
  op_map_.insert({input, std::make_shared<ngraph::op::Parameter>(
                             getType(input->dtype_), shape)});
}

}  // namespace ngraph_bridge
