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

#ifndef MXNET_NGRAPH_NGRAPH_GRAPH_H_
#define MXNET_NGRAPH_NGRAPH_GRAPH_H_

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "ngraph_graph_utils.h"

namespace ngraph_bridge {

// Useful type aliases
using NgraphNodePtr = std::shared_ptr<ngraph::Node>;
using nnvmNodePtr = std::shared_ptr<nnvm::Node>;

// Possible Types of nodes in Current Version
enum class NodeType { kVariable, kAux, kOp, kGraph, kOutput };
enum class GraphExeMode { kInfer = 0, kTrain };
constexpr int kGraphExeModeCount = static_cast<int>(GraphExeMode::kTrain) -
                                   static_cast<int>(GraphExeMode::kInfer) + 1;

// Base class for Nodes in Intermediary Analysis Graph
class Node {
 protected:
  Node(NodeType type, nnvmNodePtr node, const std::string &name)
      : type_(type),
        orig_node_(node),
        name_(name == "" ? randomString(6) : name) {}
  Node(NodeType type, nnvmNodePtr node, const std::string &name,
       const std::vector<NodePtr> &inputs)
      : type_(type),
        orig_node_(node),
        name_(name == "" ? randomString(6) : name),
        inputs_(inputs) {}

 public:
  // Function to create node label, used to export graph to graphviz for debug
  virtual std::string createNodeLabel() {
    std::ostringstream stream;
    stream << name_ << this << " [label = \"" << name_ << this << shape_
           << " \n sg=" << subgraph_ << " index=" << multi_output_index_
           << "\"];";
    return stream.str();
  }
  // basic information about node
  const NodeType type_;
  const nnvmNodePtr orig_node_;
  const std::string name_;
  std::vector<NodePtr> inputs_;

  // mxnet type information
  nnvm::TShape shape_;
  int dtype_ = 0;
  int stype_ = 0;
  mxnet::Context ctx_;

  // information to store graph parsing in
  size_t multi_output_index_ = 0;

  /// Indicates that this node is permitted to be part of a subgraph that's
  /// compiled by nGraph.
  /// The only likely reasons for leaving this false are:
  /// (a) nGraph cannot yet handle this kind of node properly, or
  /// (b) performance considerations.
  bool in_ngraph_ = false;

  std::string operation_ = "";
  int subgraph_ = 0;

  /// For debugging and logging.
  virtual void printOpDetails(std::ostream &os);
};

// Class to store Variables
// Effectivly just a wrapper for setting Node Type
class VariableNode : public Node {
 public:
  // Overloaded constructors for ease of use
  VariableNode(nnvmNodePtr node, const std::string &name)
      : Node(NodeType::kVariable, node, name) {}
  VariableNode(nnvmNodePtr node, const std::string &name,
               const std::vector<NodePtr> &inputs)
      : Node(NodeType::kVariable, node, name, inputs) {}
};

// Class to store Auxillary Tensors, mostly for Batch Norm
// Effectivly just a wrapper for setting Node Type
class AuxNode : public Node {
 public:
  // Overloaded constructors for ease of use
  AuxNode(nnvmNodePtr node, const std::string &name)
      : Node(NodeType::kAux, node, name) {}
  AuxNode(nnvmNodePtr node, const std::string &name,
          const std::vector<NodePtr> &inputs)
      : Node(NodeType::kAux, node, name, inputs) {}
};

// Node for storing operations
class OpNode : public Node {
 public:
  // Include operation in graphviz
  std::string createNodeLabel() override {
    std::ostringstream stream;
    stream << name_ << this << " [label=\"" << name_ << this
           << "\nOp: " << operation_ << shape_ << " sg=" << subgraph_ << "\"";
    auto out = stream.str();
    if (in_ngraph_) out += ", fillcolor = red, style = filled";
    out += "];";
    return out;
  }

  // Overloaded constructors for ease of use
  OpNode(nnvmNodePtr node, const std::string &name,
         const std::string &operation)
      : Node(NodeType::kOp, node, name) {
    operation_ = operation;
  }
  OpNode(nnvmNodePtr node, const std::string &name,
         const std::string &operation, const std::vector<NodePtr> &inputs)
      : Node(NodeType::kOp, node, name, inputs) {
    operation_ = operation;
  }
};

extern std::unordered_map<std::string,
                          std::shared_ptr<ngraph::runtime::Backend>>
    backends;

inline std::string get_backend_name(const mxnet::Context &context) {
#if MXNET_USE_CUDA
  if (context.dev_type == mxnet::Context::kGPU) {
    return "GPU";
  }
#endif
  // user specified ngraph backend
  if (context.dev_type == mxnet::Context::kNGraph) {
    auto backend = NGraphContextFromDevID(context.dev_id);
    return backend.first + ":" + std::to_string(backend.second);
  }
  // "CPU" is fallback backend
  return "CPU";
}

inline std::shared_ptr<ngraph::runtime::Backend> GetBackendFromContext(
    const mxnet::Context &context) {
  auto backend_key = get_backend_name(context);
  if (backends.count(backend_key) == 0) {
    auto backend = ngraph::runtime::Backend::create(backend_key);
    backends[backend_key] = backend;
  }
  return backends[backend_key];
}

class OutputElement;

using MapEntry = std::pair<nnvmNodePtr, size_t>;
/*
Graph class
Graph subclasses Node so that we can embed graphs into other graphs
This is useful when we take a graph and replace it with an ngraph computation
TODO: Refactor into Graph and subgraph?
*/
class Graph : public Node {
 public:
  // Graph with optional fprop cache
  Graph(const std::string &name = "",
        const mxnet::Context &context = mxnet::Context::CPU(),
        const bool enable_fprop_cache = true)
      : Node(NodeType::kGraph, nullptr, name),
        context_(context),
        enable_fprop_cache(enable_fprop_cache) {
    fprop_cache = std::make_shared<ngraph::FpropCache>();
    is_reuse_mem = context.dev_type == mxnet::Context::kNGraph ? false : true;
  }

  ~Graph() {
    // Clean up nGraph's compilation cache so we don't have a memory leak
    auto backend = GetBackendFromContext(context_);
    for (int i = 0; i < kGraphExeModeCount; ++i) {
      if (ngraph_forward[i]) {
        backend->remove_compiled_function(ngraph_forward[i]);
      }
      if (ngraph_backward[i]) {
        backend->remove_compiled_function(ngraph_backward[i]);
      }
    }
  }

  std::string createNodeLabel() override {
    std::ostringstream stream;
    stream << name_ << this << " [label = \"" << name_ << this << shape_
           << " \n sg=" << subgraph_ << " index=" << multi_output_index_
           << "\", fillcolor = green, style = filled];";
    return stream.str();
  }

  // TODO(mbrookhart): We're carrying a map of NodeEntry to bridge nodes
  // for easy lookup during parsing. This relies on nodes being added through
  // the AddNode method, but nodes_ is a public variable, so this isn't 100%
  // seafe probably need a refactor into a getter/setter type situation.

  // Add a node to the graph
  void AddNode(NodePtr node) {
    entry_map_[MapEntry{node->orig_node_, node->multi_output_index_}] = node;
    nodes_.emplace_back(node);
  }

  // get the node corresponding to an orig_node
  NodePtr operator[](const nnvm::NodeEntry &entry) {
    MapEntry tmp{entry.node, entry.index};
    if (entry_map_.count(tmp)) {
      return entry_map_[tmp];
    }
    return nullptr;
  }

  bool forward_train_computed{false};
  size_t num_outputs_ = 1;
  size_t num_adjoints_ = 0;
  // nodes in this graph
  std::vector<NodePtr> nodes_;
  std::map<MapEntry, NodePtr> entry_map_;
  // functions to execute this graph in ngraph.
  // Note: ngraph_backward[GraphExeMode::kInfer] should always be null, but we
  // define it for consisteny.
  std::shared_ptr<ngraph::Function> ngraph_forward[kGraphExeModeCount];
  std::shared_ptr<ngraph::Function> ngraph_backward[kGraphExeModeCount];
  std::shared_ptr<ngraph::FpropCache> fprop_cache;

  const mxnet::Context context_;
  std::vector<std::shared_ptr<ngraph::runtime::TensorView>>
      cached_values[kGraphExeModeCount];
  std::vector<std::shared_ptr<ngraph::runtime::TensorView>>
      cached_aux_values[kGraphExeModeCount];

  std::vector<int> cached_aux_positions[kGraphExeModeCount];

  const bool enable_fprop_cache;

  std::vector<NodePtr> outputs_;
  std::vector<std::shared_ptr<OutputElement>> output_elements_;
  std::vector<bool> input_is_weight_;
  bool zero_grad = false;
  // is loss is used to mark graphs as ending in loss layers to
  // handle some zero_grad errors with batch_take
  bool is_loss = false;
  bool is_reuse_mem = true;
};

// Element to represent outputs of Graph objects embedded in other Graph objects
class OutputElement : public Node {
 public:
  OutputElement(std::shared_ptr<Graph> node, size_t index)
      : Node(NodeType::kOutput, node->outputs_[index]->orig_node_,
             node->outputs_[index]->name_),
        base_node_(node->outputs_[index]) {
    shape_ = base_node_->shape_;
    dtype_ = base_node_->dtype_;

    inputs_.push_back(node);

    multi_output_index_ = index;
    subgraph_ = base_node_->subgraph_;
  }

  std::string createNodeLabel() override {
    std::ostringstream stream;
    stream << name_ << this << " [label = \"" << name_ << this << shape_
           << " \n sg=" << subgraph_ << " index=" << multi_output_index_
           << "\", fillcolor = purple, style = filled];";
    return stream.str();
  }

  NodePtr base_node_;
};

/**
 * High level function that does the subgraph identification
 */
void IdentifySubgraphs(Graph *graph, const std::function<bool(NodePtr)> &func);

/**
 * Convert graph from identified nodes to a network of nodes and graphs,
 * each graph node represented a combined ngraph operation
 */
void CollapseSubgraph(Graph *graph, int subgraph_num);

/**
 * Selection of nodes based on function criterion.
 * Note: uses DFSUtil().
 */
std::vector<NodePtr> SelectNodes(NodePtr node,
                                 const std::function<bool(NodePtr)> &func);

/**
 * Finds simply connected ngraph operations
 */
std::vector<NodePtr> FindSubgraph(const Graph &graph, NodePtr node,
                                  const std::function<bool(NodePtr)> &func);

// Struct containing functors used as a utility for traversing a graph
struct GraphVisitor {
  std::function<void(NodePtr)> operation;
  std::function<bool(NodePtr, NodePtr)> stop_condition;
  std::function<std::vector<NodePtr>(NodePtr)> get_inputs = [](NodePtr n) {
    return n->inputs_;
  };
};

// Perform a DFS graph traversal non-recursively but always ensuring
// that the inputs to a node are operated on before the node. It also checks
// for graph cycles and throws an error if they are found.
void GraphTraverse(NodePtr node, const GraphVisitor &visitor);

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_GRAPH_H_
