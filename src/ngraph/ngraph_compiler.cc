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

#include <mshadow/base.h>

#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "../executor/exec_pass.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "nnvm/tuple.h"

namespace ngraph_bridge {

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodeEntry CreateNNVMNode(std::shared_ptr<Graph> subgraph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = subgraph->name_;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(subgraph);
  // setup the ninputs to the node
  for (auto input : subgraph->inputs_)
    node->inputs.emplace_back(nnvm::NodeEntry{input->orig_node_, 0, 0});
  // create dummy node parameters
  NGraphParam op;
  node->attrs.parsed = std::move(op);

  // init and return NodeEntry
  return nnvm::NodeEntry{node, 0, 0};
}

// Generator to create functions that convert mxnet layer operations
// into a series of ngraph operations
LayerGraphs create_layer_graphs() {
  LayerGraphs layer_funcs;

  // Split is an operation with multiple outputs that splits
  // a tensor into multiple tensors by creating even slices along
  // one axis. To ease the interface with ngraph, we convert split into
  // a slice based subgraph.
  layer_funcs[std::string("split")] = [](const NodePtr node) {
    Graph tmpGraph(node->name_);
    int num_outputs = 1;
    for (auto& kv : node->orig_node_->attrs.dict)
      if (kv.first == "num_outputs") num_outputs = std::stoi(kv.second);

    tmpGraph.num_outputs = num_outputs;
    for (int i = 0; i < num_outputs; ++i) {
      auto tmpslice = std::make_shared<OpNode>(
          node->orig_node_, node->name_ + "_" + std::to_string(i), "split");
      tmpslice->inputs_.push_back(node->inputs_[0]);
      tmpslice->multi_output_index_ = i;
      tmpGraph.AddNode(tmpslice);
    }

    return tmpGraph;
  };

  // Slice channel is an alias for split
  layer_funcs[std::string("SliceChannel")] = layer_funcs["split"];

  return layer_funcs;
}

// infer nnvm::Graph shape and dtype for bind case
// reused from GraphExecutor::Init in graph_executor.cc
void Compiler::Infer(const BindArg* bind) {
  const auto& idx = graph_.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < bind->kNumForwardInputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      shapes_.push_back(bind->aux_states_[aux_top].shape());
      dtypes_.push_back(bind->aux_states_[aux_top].dtype());
      ++aux_top;
    } else {
      shapes_.push_back(bind->in_args_[arg_top].shape());
      dtypes_.push_back(bind->in_args_[arg_top].dtype());
      ++arg_top;
    }
  }

  // append default shapes / dtypes so that vector size = graph size
  shapes_.resize(idx.input_nodes().size(), nnvm::TShape());
  dtypes_.resize(idx.input_nodes().size(), -1);
}

// infer nnvm::Graph shape and dtype for simple bind case
// reused from GraphExecutor::Init in graph_executor.cc
void Compiler::Infer(const SimpleBindArg* simplebind) {
  const auto& idx = graph_.indexed_graph();
  shapes_.resize(idx.input_nodes().size(), nnvm::TShape());
  dtypes_.resize(idx.input_nodes().size(), -1);
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < simplebind->kNumForwardInputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it1 = simplebind->shape_map_.find(name);
    if (simplebind->shape_map_.end() != it1) {
      shapes_[i] = it1->second;
    }
    auto it2 = simplebind->dtype_map_.find(name);
    if (simplebind->dtype_map_.end() != it2) {
      dtypes_[i] = it2->second;
    }
  }
}

// Compiler initialization
Compiler::Compiler(const mxnet::Context& context)
    : ngraph_("ngraph_" + randomString(6), context) {}

Compiler::Compiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
                   const NNVMNodeVec& inputs, const BindArgBase& bindbase,
                   const mxnet::Context& context)
    : ngraph_("ngraph_" + randomString(6), context) {
  DeepCopy(graph);

  // infer nnvm::Graph shape and type
  auto bind = dynamic_cast<const BindArg*>(&bindbase);
  auto simplebind = dynamic_cast<const SimpleBindArg*>(&bindbase);
  if (bind != nullptr) {
    Infer(bind);
  } else if (simplebind != nullptr) {
    Infer(simplebind);
  }
  MakeCopiedInputs(inputs);
  ProcessGraph(feed_dict);
}

void Compiler::ProcessGraph(const NDArrayMap& feed_dict) {
  graph_ = mxnet::exec::InferShape(std::move(graph_), std::move(shapes_),
                                   "__shape__");
  // TODO(adstraw): may or may not need error checking
  // if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
  //  HandleInferShapeError(num_forward_inputs, g.indexed_graph(),
  //    g.GetAttr<nnvm::ShapeVector>("shape"));
  //}

  graph_ = mxnet::exec::InferType(std::move(graph_), std::move(dtypes_),
                                  "__dtype__");
  // TODO(adstraw): may or may not need error checking
  // if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
  //  HandleInferTypeError(num_forward_inputs, g.indexed_graph(),
  //    g.GetAttr<nnvm::DTypeVector>("dtype"));
  //}

  MakeCopiedFeedDict(feed_dict);
  ParseNnvmGraph();
  CheckInNgraph();

  IdentifySubgraphs(ngraph_, [&feed_dict](NodePtr s) -> bool {
    bool in_feed_dict = false;
    for (auto kv : feed_dict) {
      if (kv.first.node->attrs.name == s->name_) {
        in_feed_dict = true;
        break;
      }
    }
    return (s->in_ngraph_ && s->type_ == NodeType::kOp && !in_feed_dict);
  });
}

// Main compilation function
nnvm::Graph Compiler::Compile() {
  // Output Graphviz dot files (pre collapse) for vizualization
  if (false) WriteSubgraphDots(ngraph_, "pre_collapse");

  CollapseSubgraphs(&ngraph_);

  // Output Graphviz dot files (post collapse) for vizualization
  if (false) WriteSubgraphDots(ngraph_, "post_collapse");

  for (auto node : ngraph_.nodes_) {
    // store the input variable shape for use by nnvm
    // This is happening because my nnvm graph manipulations are
    // breaking the infer shape functionality, so shapes of inputs
    // don't get properly inferred. Works, because we're inferring
    // the shapes before doing all of this, but not ideal
    if (node->type_ == NodeType::kAux || node->type_ == NodeType::kVariable) {
      ngraph_shape_[node->name_] = node->shape_;
      ngraph_dtype_[node->name_] = node->dtype_;
    }
  }

  // find the subgraphs
  for (auto n : ngraph_.nodes_) {
    if (n->type_ == NodeType::kGraph) {
      // extract and compile subgraph
      auto sg = compiler_.Compile(n);
      // register compiled subgraph with nnvm
      register_subgraph(sg);
      // create nnvm node
      auto sg_node = CreateNNVMNode(sg);

      auto matches = [&sg](nnvm::NodeEntry n) -> bool {
        return (n.node == sg->nodes_.back()->orig_node_) &&
               (n.index == sg->nodes_.back()->multi_output_index_);
      };

      // Replace outputs if needed
      for (auto& output : graph_.outputs)
        if (matches(output)) output = sg_node;

      // use nnvm depth first search to fix node connections in nnvm
      nnvm::DFSVisit(
          graph_.outputs, [sg_node, &matches](const nnvm::NodePtr node) {
            for (auto input : node->inputs) {
              auto it = std::find_if(node->inputs.begin(), node->inputs.end(),
                                     matches);

              if (it != node->inputs.end()) {
                node->inputs.insert(it, sg_node);
                node->inputs.erase(std::remove_if(node->inputs.begin(),
                                                  node->inputs.end(), matches),
                                   node->inputs.end());
              }
            }
          });
    }
  }

  // create a new output graph
  nnvm::Graph out_graph;

  // initialize it with original graph nodes
  out_graph.outputs = graph_.outputs;

  return std::move(out_graph);
}

// create copied saved states
StateMap Compiler::CopySavedStates(const StateMap& saved_states) {
  StateMap new_saved_states;
  for (auto kv : saved_states) {
    new_saved_states[node_map_[kv.first].get()] = kv.second;
  }
  return new_saved_states;
}

// create a copied feed dict
void Compiler::MakeCopiedFeedDict(const NDArrayMap& feed_dict) {
  for (auto kv : feed_dict) {
    auto e = kv.first;
    e.node = node_map_[kv.first.node.get()];
    feed_dict_[e] = kv.second;
  }
}

// create a vector of copied inputs
void Compiler::MakeCopiedInputs(const NNVMNodeVec& inputs) {
  for (auto node : inputs) {
    inputs_.push_back(node_map_[node.get()]);
  }
}

void Compiler::CopyNodes(const nnvm::Graph& graph) {
  nnvm::DFSVisit(graph.outputs, [this](const nnvm::NodePtr node) {
    if (!node_map_.count(node.get()))
      node_map_[node.get()] = std::make_shared<nnvm::Node>(*node);
  });
}

void Compiler::DeepCopy(const nnvm::Graph& graph) {
  // make copies of all the graph nodes
  CopyNodes(graph);
  // reset the inputs of the copies
  for (auto kv : node_map_)
    for (auto& input : kv.second->inputs)
      input.node = node_map_[input.node.get()];

  // set the output graph to use the copied nodes
  graph_.outputs = graph.outputs;
  for (auto& out : graph_.outputs) out.node = node_map_[out.node.get()];
}

// Check nodes in NGraph
void Compiler::CheckInNgraph() {
  for (auto node : ngraph_.nodes_) {
    if (node->type_ == NodeType::kOp) {
      if (compiler_.ngraph_op_funcs_[kInfer].count(node->operation_)) {
        node->in_ngraph_ = true;
        if (node->dtype_ == mshadow::kFloat16) {
          node->in_ngraph_ = false;
        } else {
          for (auto input : node->inputs_) {
            if (input->dtype_ == mshadow::kFloat16) {
              node->in_ngraph_ = false;
            }
          }
        }
      }
    }
  }
}

// Function that parses an nnvm Graph into an intermediary graph
void Compiler::ParseNnvmGraph() {
  // Create dictionary of layer->ngraph functions
  auto layer_funcs = create_layer_graphs();
  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph_.outputs, [this,
                                  &layer_funcs](const nnvm::NodePtr node) {
    const auto& idx = this->graph_.indexed_graph();

    const auto& mutable_nodes = idx.mutable_input_nodes();
    const uint32_t nid = idx.node_id(node.get());
    if (mutable_nodes.count(nid) != 0) {
      // add an auxillary node to the graph
      this->ngraph_.AddNode(std::make_shared<AuxNode>(node, node->attrs.name));
    } else if (node->is_variable()) {
      // add variable to the graph
      this->ngraph_.AddNode(
          std::make_shared<VariableNode>(node, node->attrs.name));
    } else {
      // create operation node
      auto op_name = clean_opname(node->op()->name);
      auto op_node = std::make_shared<OpNode>(node, node->attrs.name, op_name);
      // setup operation inputs
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = node->inputs[i];
        std::shared_ptr<Node> tmpnode;
        try {
          tmpnode = this->ngraph_[e];
        } catch (char const* error) {
          try {
            tmpnode = this->ngraph_[e];
          } catch (char const* error) {
            tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
            this->ngraph_.AddNode(tmpnode);
          }
        }
        op_node->inputs_.emplace_back(tmpnode);
      }

      // function that renames or expands things like activation and
      // split
      auto expand_layers = [this,
                            &layer_funcs](std::shared_ptr<OpNode>& op_node) {
        auto tmp = layer_funcs[op_node->operation_](op_node);
        if (tmp.num_outputs > 1)
          this->ngraph_.nodes_.erase(
              std::remove(this->ngraph_.nodes_.begin(),
                          this->ngraph_.nodes_.end(), op_node),
              this->ngraph_.nodes_.end());

        for (auto n : tmp.nodes_) this->ngraph_.AddNode(n);
      };

      if (layer_funcs.count(op_node->operation_) != 0) {
        // perform layer expansions
        expand_layers(op_node);
      } else {
        // add operation
        this->ngraph_.AddNode(op_node);
      }
    }
  });

  // get the shape and data types of all of the nodes
  const auto& idx = graph_.indexed_graph();
  const auto inferred_shapes =
      graph_.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = graph_.GetAttr<std::vector<int>>("dtype");
  for (auto node : this->ngraph_.nodes_) {
    const uint32_t nid = idx.node_id(node->orig_node_.get());
    const uint32_t eid = idx.entry_id(nid, 0);
    node->shape_ = inferred_shapes[eid];
    node->dtype_ = inferred_dtypes[eid];
  }
}

}  // namespace ngraph_bridge
