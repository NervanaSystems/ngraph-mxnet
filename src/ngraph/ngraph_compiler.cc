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

#include <mshadow/base.h>

#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include <atomic>
#include <sstream>
#include <thread>
#include "../executor/exec_pass.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_stats.h"
#include "ngraph_utils.h"
#include "nnvm/tuple.h"

namespace ngraph_bridge {

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodePtr CreateNNVMNode(std::shared_ptr<Graph> subgraph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = subgraph->name_;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(subgraph);
  // setup the inputs to the node
  for (auto input : subgraph->inputs_)
    if (input->type_ == NodeType::kOutput) {
      auto n = std::dynamic_pointer_cast<OutputElement>(input);
      node->inputs.emplace_back(nnvm::NodeEntry{
          n->base_node_->orig_node_,
          static_cast<uint32_t>(n->base_node_->multi_output_index_),
          static_cast<uint32_t>(0)});

    } else {
      node->inputs.emplace_back(nnvm::NodeEntry{
          input->orig_node_, static_cast<uint32_t>(input->multi_output_index_),
          static_cast<uint32_t>(0)});
    }
  // create dummy node parameters
  NGraphParam op;
  op.g = subgraph;

  node->attrs.parsed = std::move(op);
  return node;
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
      stypes_.push_back(bind->aux_states_[aux_top].storage_type());
      ++aux_top;
    } else {
      shapes_.push_back(bind->in_args_[arg_top].shape());
      dtypes_.push_back(bind->in_args_[arg_top].dtype());
      stypes_.push_back(bind->in_args_[arg_top].storage_type());
      ++arg_top;
    }
  }

  // append default shapes / dtypes so that vector size = graph size
  shapes_.resize(idx.input_nodes().size(), nnvm::TShape());
  dtypes_.resize(idx.input_nodes().size(), -1);
  stypes_.resize(idx.input_nodes().size(), mxnet::kUndefinedStorage);
}

// infer nnvm::Graph shape and dtype for simple bind case
// reused from GraphExecutor::Init in graph_executor.cc
void Compiler::Infer(const SimpleBindArg* simplebind) {
  const auto& idx = graph_.indexed_graph();
  shapes_.resize(idx.input_nodes().size(), nnvm::TShape());
  dtypes_.resize(idx.input_nodes().size(), -1);
  stypes_.resize(idx.input_nodes().size(), mxnet::kUndefinedStorage);
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
    auto it3 = simplebind->stype_map_.find(name);
    if (simplebind->stype_map_.end() != it3) {
      stypes_[i] = it3->second;
    }
  }
}

static std::atomic<int> graph_counter(0);

std::string get_ngraph_name() {
  std::stringstream name;
  name << "ngraph_" << std::this_thread::get_id() << "_" << graph_counter++;
  return name.str();
}

// Compiler initialization with fprop cache disabled
Compiler::Compiler(const mxnet::Context& context)
    : ngraph_(get_ngraph_name(), context, false) {}

// Compiler initialization for gluon hybrid
Compiler::Compiler(const nnvm::Graph& graph, const mxnet::Context& context,
                   const std::vector<nnvm::TShape>& shapes,
                   const std::vector<int>& dtypes,
                   const std::vector<int>& stypes)
    : ngraph_(get_ngraph_name(), context),
      shapes_(shapes),
      dtypes_(dtypes),
      stypes_(stypes) {
  DeepCopy(graph);
  graph_.attrs["context"] = std::make_shared<nnvm::any>(
      mxnet::exec::ContextVector(graph_.indexed_graph().num_nodes(), context));
  ProcessGraph({});
}
// Compiler initialization
Compiler::Compiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
                   const NNVMNodeVec& inputs, const BindArgBase& bindbase,
                   const mxnet::Context& context)
    : ngraph_(get_ngraph_name(), context) {
  DeepCopy(graph);
  graph_.attrs["context"] = std::make_shared<nnvm::any>(
      mxnet::exec::ContextVector(graph_.indexed_graph().num_nodes(), context));
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

  graph_ = mxnet::exec::InferStorageType(std::move(graph_), std::move(stypes_),
                                         "__storage_type__");
  MakeCopiedFeedDict(feed_dict);
  ParseNnvmGraph();
  CheckInNgraph();
}

void Compiler::IdentifyCollapseGraphs() {
  if (ngraph_log_verbose()) {
    std::cout << "NGRAPH_BRIDGE: processing " << ngraph_.name_ << std::endl;
  }
  // Output Graphviz dot files (pre collapse) for vizualization
  if (ngraph_log_viz()) {
    WriteSubgraphDots(ngraph_, ngraph_.name_ + "_pre_collapse");
  }

  IdentifySubgraphs(&ngraph_, [this](NodePtr s) -> bool {
    bool in_feed_dict = false;
    for (auto kv : feed_dict_) {
      if (kv.first.node->attrs.name == s->name_) {
        in_feed_dict = true;
        break;
      }
    }
    return (s->in_ngraph_ && s->type_ == NodeType::kOp && !in_feed_dict);
  });

  // Output Graphviz dot files (post collapse) for vizualization
  if (ngraph_log_viz()) {
    WriteSubgraphDots(ngraph_, ngraph_.name_ + "_post_collapse");
  }
}

void Compiler::CreateSubgraphNNVMNodes() {
  // find the subgraphs
  for (auto n : ngraph_.nodes_) {
    if (n->type_ == NodeType::kGraph) {
      // extract and compile subgraph
      compiler_.setExeMode(GraphExeMode::kInfer);
      auto sg = compiler_.Compile(n);

      // compile subgraph in other execution modes,
      for (int i = 1; i < kGraphExeModeCount; ++i) {
        // set graph execution mode
        compiler_.setExeMode(static_cast<GraphExeMode>(i));
        compiler_.Compile(n);
      }

      // register compiled subgraph with nnvm
      register_subgraph(sg);
	  
	  // add subgraph to stats tracker
      if (ngraph_log_timer()) {
        NGraphStats::get_instance().add(sg);
      }

      // create nnvm node
      auto node = CreateNNVMNode(sg);
      compiled_nodes_.insert({sg, node});
    }
  }
}

struct NodeEntryHash {
  size_t operator()(const nnvm::NodeEntry& key) const {
    size_t hash = std::hash<nnvm::NodePtr>()(key.node);
    hash = hash_combine(hash, key.index);
    hash = hash_combine(hash, key.version);
    return hash;
  }
};

struct NodeEntryEqual {
  bool operator()(const nnvm::NodeEntry& lhs,
                  const nnvm::NodeEntry& rhs) const {
    return ((lhs.node == rhs.node) && (lhs.index == rhs.index) &&
            (lhs.version == rhs.version));
  }
};

void Compiler::ConnectSubgraphNodes() {
  // create a map of original NodeEntries -> subgraph output NodeEntries
  std::unordered_map<nnvm::NodeEntry, nnvm::NodeEntry, NodeEntryHash,
                     NodeEntryEqual>
      out_map;
  for (auto kv : compiled_nodes_) {
    for (auto output : kv.first->output_elements_) {
      nnvm::NodeEntry orig_entry{
          output->base_node_->orig_node_,
          static_cast<uint32_t>(output->base_node_->multi_output_index_),
          static_cast<uint32_t>(0)};
      nnvm::NodeEntry new_entry{
          kv.second, static_cast<uint32_t>(output->multi_output_index_),
          static_cast<uint32_t>(0)};
      out_map[orig_entry] = new_entry;
    }
  }
  // replace inputs in all of the graphs if they are now outputs of other
  // subgraphs
  for (auto kv : compiled_nodes_) {
    for (auto& input : kv.second->inputs) {
      if (out_map.count(input) != 0) {
        input = out_map[input];
      }
    }
  }
}

void Compiler::CollapseNNVMGraph() {
  for (auto n : ngraph_.nodes_) {
    // Find the subgraphs
    if (n->type_ == NodeType::kGraph) {
      auto sg = std::dynamic_pointer_cast<Graph>(n);
      // get the NNVM node
      auto node = compiled_nodes_.at(sg);
      // Loop over the output elements, and replace NNVM node entries
      // with output Node Entries
      for (auto output : sg->output_elements_) {
        nnvm::NodeEntry sg_node{
            node, static_cast<uint32_t>(output->multi_output_index_),
            static_cast<uint32_t>(0)};

        auto matches = [&output](nnvm::NodeEntry n) -> bool {
          return (n.node == output->base_node_->orig_node_) &&
                 (n.index == output->base_node_->multi_output_index_);
        };

        // Replace outputs if needed
        for (auto& nnvm_output : graph_.outputs)
          if (matches(nnvm_output)) nnvm_output = sg_node;

        // use nnvm depth first search to fix node connections in nnvm
        nnvm::DFSVisit(graph_.outputs,
                       [sg_node, &output, &matches](const nnvm::NodePtr node) {
                         for (auto input : node->inputs) {
                           auto it = std::find_if(node->inputs.begin(),
                                                  node->inputs.end(), matches);

                           if (it != node->inputs.end()) {
                             node->inputs[it - node->inputs.begin()] = sg_node;
                           } else {
                             break;
                           }
                         }
                       });
      }
    }
  }
}

void Compiler::CleanUpUneededReferences() {
  // Clean up the nodes in the subgraph that we don't need anymore
  // so we don't keep extra shared pointers around
  // this is spaghetti
  // TODO(mbrookhart): Ask DLMC for the capability to destroy nnvm::op
  // objects so we don't have to do this anymore.
  for (auto kv : compiled_nodes_) {
    for (auto input : kv.first->inputs_) {
      input->inputs_.clear();
    }
    for (auto output : kv.first->outputs_) {
      output->inputs_.clear();
    }
    for (auto output_element : kv.first->output_elements_) {
      output_element->inputs_.clear();
      output_element->base_node_ = nullptr;
    }
    kv.first->nodes_.clear();
  }
}

// Main compilation function
nnvm::Graph Compiler::Compile() {
  IdentifyCollapseGraphs();

  for (auto node : ngraph_.nodes_) {
    // store the input variable shape for use by nnvm
    // This is happening because my nnvm graph manipulations are
    // breaking the infer shape functionality, so shapes of inputs
    // don't get properly inferred. Works, because we're inferring
    // the shapes before doing all of this, but not ideal
    if (node->type_ == NodeType::kAux || node->type_ == NodeType::kVariable) {
      ngraph_shape_[node->name_] = node->shape_;
      ngraph_dtype_[node->name_] = node->dtype_;
      ngraph_stype_[node->name_] = node->stype_;
    }
  }

  CreateSubgraphNNVMNodes();
  ConnectSubgraphNodes();
  CollapseNNVMGraph();
  CleanUpUneededReferences();
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
  std::unordered_set<std::string> unsupported_op_names;
  for (const std::shared_ptr<ngraph_bridge::Node>& node : ngraph_.nodes_) {
    // The bridge code only has nGraph emitters for kOp-type nodes.
    if (node->type_ == NodeType::kOp) {
      if (compiler_.ngraph_op_funcs_.count(node->operation_)) {
        node->in_ngraph_ = true;

        if (node->operation_ == "BatchNorm") {
          auto shape = TShape_to_NShape(node->inputs_[0]->shape_);
          if (shape[1] % 8 != 0) {
            // MXNet outperforms nGraph in this case.
            node->in_ngraph_ = false;
          }
        } else if (node->operation_ == "LeakyReLU") {
          // We haven't yet implemented all activation functions for
          // LeaklyReLU...
          const std::string act_type =
              get_default(node, "act_type", std::string("leaky"));
          if (act_type != "leaky") {
            node->in_ngraph_ = false;
          }
        }

        // nGraph doesn't yet support float16.
        if (node->dtype_ == mshadow::kFloat16 ||
            node->stype_ != mxnet::kDefaultStorage) {
          node->in_ngraph_ = false;
        } else {
          for (auto input : node->inputs_) {
            if (input->dtype_ == mshadow::kFloat16 ||
                input->stype_ != mxnet::kDefaultStorage) {
              node->in_ngraph_ = false;
            }
          }
        }
      } else {
        if (ngraph_log_verbose()) {
          unsupported_op_names.insert(node->operation_);
        }

        if (ngraph_log_verbose_detail()) {
          std::cout << "NGRAPH_BRIDGE: Unsupported Op instance (verbose):"
                    << std::endl;
          node->printOpDetails(std::cout);
          std::cout << std::endl;
        }
      }
    }
  }
  for (const auto& name : unsupported_op_names) {
    std::cout << "NGRAPH_BRIDGE: Unsupported Op: " << name << std::endl;
  }
}

// Function that parses an nnvm Graph into an intermediary graph
void Compiler::ParseNnvmGraph() {
  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph_.outputs, [this](const nnvm::NodePtr node) {
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

      // If it's a multi output op (and not Batchnorm)
      // replace it with a set of nodes that correspond to each output of
      // the op
      // TODO(mbrookhart): Handle this more carefully somehow?
      // not sure how many ops there actually are that need multi-output
      if (node->num_outputs() > 1 && op_name != "BatchNorm") {
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          auto tmpop = std::make_shared<OpNode>(
              op_node->orig_node_, op_node->name_ + "_" + std::to_string(i),
              op_node->operation_);

          for (auto input : op_node->inputs_) tmpop->inputs_.push_back(input);
          tmpop->multi_output_index_ = i;

          this->ngraph_.AddNode(tmpop);
        }
      } else {
        // add operation
        this->ngraph_.AddNode(op_node);
      }
    }
  });

  // set up the inputs to all of the  nodes in the graph
  for (auto node : ngraph_.nodes_) {
    for (size_t i = 0; i < node->orig_node_->inputs.size(); ++i) {
      const nnvm::NodeEntry& e = node->orig_node_->inputs[i];
      std::shared_ptr<Node> tmpnode;
      tmpnode = this->ngraph_[e];
      if (tmpnode == nullptr) {
        throw std::runtime_error(
            "NGRAPH_BRIDGE: couldn't parse the NNVM graph");
      }
      node->inputs_.emplace_back(tmpnode);
    }
  }

  // set up the outputs to the parsed bridge graph
  for (auto e : graph_.outputs) {
    ngraph_.outputs_.push_back(ngraph_[e]);
  }

  // get the shape, data and storage types of all of the nodes
  const auto& idx = graph_.indexed_graph();
  const auto inferred_shapes =
      graph_.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = graph_.GetAttr<std::vector<int>>("dtype");
  const auto& inferred_stypes =
      graph_.GetAttr<mxnet::StorageTypeVector>("storage_type");
  for (auto node : this->ngraph_.nodes_) {
    const uint32_t nid = idx.node_id(node->orig_node_.get());
    const uint32_t eid = idx.entry_id(nid, node->multi_output_index_);
    node->shape_ = inferred_shapes[eid];
    node->dtype_ = inferred_dtypes[eid];
    node->stype_ = inferred_stypes[eid];
  }
}

}  // namespace ngraph_bridge
