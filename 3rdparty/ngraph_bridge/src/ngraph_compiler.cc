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
#include <cstdlib>
#include <sstream>
#include <thread>
#include "../../../src/executor/exec_pass.h"
#include "../../../src/imperative/imperative_utils.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_sgcompiler_utils.h"
#include "ngraph_stats.h"
#include "ngraph_utils.h"
#include "nnvm/tuple.h"

namespace ngraph_bridge {

// TODO(mbrookhart): remove when DEX becomes default
static int ngraph_dex = setenv("NGRAPH_DEX", "1", true);

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodePtr CreateNNVMNode(std::shared_ptr<Graph> subgraph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = subgraph->name_;
  // get the registered operation for the node
  node->attrs.op = nnvm::Op::Get("_ngraph_subgraph_op");
  // setup the inputs to the node
  for (auto input : subgraph->inputs_)
    if (input->type_ == NodeType::kOutput && subgraph->subgraph_ > 0) {
      auto n = std::dynamic_pointer_cast<OutputElement>(input);
      node->inputs.emplace_back(nnvm::NodeEntry{
          n->base_node_->orig_node_,
          static_cast<uint32_t>(n->base_node_->multi_output_index_),
          static_cast<uint32_t>(0)});
      // } else if (input->type_ == NodeType::kOutput) {
      //   auto n = std::dynamic_pointer_cast<OutputElement>(input);
      //   node->inputs.emplace_back(nnvm::NodeEntry{
      //       n->base_node_->orig_node_,
      //       static_cast<uint32_t>(n->multi_output_index_),
      //       static_cast<uint32_t>(0)});
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

static std::atomic<int> graph_counter(1);

std::string get_ngraph_name() {
  std::stringstream name;
  name << "ngraph_" << std::this_thread::get_id() << "_" << graph_counter++;
  return name.str();
}

// Compiler initialization with fprop cache disabled
Compiler::Compiler(const mxnet::Context& context)
    : ngraph_(get_ngraph_name(), context, false) {}

Compiler::Compiler(const nnvm::Graph& graph, const NNVMNodeVec& symbol_inputs,
                   const std::vector<mxnet::NDArray*>& inputs)
    : ngraph_(get_ngraph_name(), inputs[0]->ctx()) {
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    shapes_.emplace_back(inputs[i]->shape());
    dtypes_.emplace_back(inputs[i]->dtype());
    stypes_.emplace_back(inputs[i]->storage_type());
  }

  DeepCopy(graph);
  graph_.attrs["context"] =
      std::make_shared<nnvm::any>(mxnet::exec::ContextVector(
          graph_.indexed_graph().num_nodes(), inputs[0]->ctx()));
  MakeCopiedInputs(symbol_inputs);
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
    return (s->in_ngraph_ && !in_feed_dict);
  });

  // Output Graphviz dot files (post collapse) for vizualization
  if (ngraph_log_viz()) {
    WriteSubgraphDots(ngraph_, ngraph_.name_ + "_post_collapse");
  }
}

void Compiler::CreateSubgraphNNVMNodes() {
  // find the subgraphs
  for (auto n : ngraph_.nodes_) {
    if (n->type_ == NodeType::kGraph && n->subgraph_ > 0) {
      // extract and compile subgraph
      compiler_.setExeMode(GraphExeMode::kInfer);
      auto sg = compiler_.Compile(n);

      // compile subgraph in other execution modes,
      for (int i = 1; i < kGraphExeModeCount; ++i) {
        // set graph execution mode
        compiler_.setExeMode(static_cast<GraphExeMode>(i));
        compiler_.Compile(n);
      }

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
    if (n->type_ == NodeType::kGraph && n->subgraph_ > 0) {
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
    kv.first->entry_map_.clear();
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

nnvm::Graph Compiler::GetCachedOpGraph(
    const std::vector<mxnet::NDArray*>& inputs) {
  nnvm::ShapeVector shape_inputs;
  nnvm::DTypeVector dtype_inputs;
  std::vector<int> storage_type_inputs;
  shape_inputs.reserve(inputs.size());
  dtype_inputs.reserve(inputs.size());
  storage_type_inputs.reserve(inputs.size());
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    shape_inputs.emplace_back(inputs[i]->shape());
    dtype_inputs.emplace_back(inputs[i]->dtype());
    storage_type_inputs.emplace_back(inputs[i]->storage_type());
  }
  // create a new output graph
  auto g = Compile();
  mxnet::imperative::CheckAndInferShape(&g, std::move(shape_inputs), true);
  mxnet::imperative::CheckAndInferType(&g, std::move(dtype_inputs), true);
  exec::DevMaskVector cached_dev_mask(g.indexed_graph().num_nodes(),
                                      inputs[0]->ctx().dev_mask());
  mxnet::imperative::CheckAndInferStorageType(
      &g, std::move(cached_dev_mask), std::move(storage_type_inputs), true);

  const auto& idx = g.indexed_graph();
  std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
  for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
  for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
  }
  g.attrs["forward_ref_count"] = std::make_shared<dmlc::any>(ref_count);
  return std::move(g);
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

bool bad_type(const NodePtr& node) {
  return node->dtype_ == mshadow::kFloat16 ||
         node->dtype_ == mshadow::kFloat64 ||
         node->stype_ != mxnet::kDefaultStorage;
}

bool Compiler::IsInNGraph(const NodePtr& node) {
  bool in_ngraph_ = false;
  if (bad_type(node)) {
    return false;
  } else if (std::find_if(node->inputs_.begin(), node->inputs_.end(),
                         bad_type) != node->inputs_.end()) {
    return false;
  } else if (node->type_ == NodeType::kOp || node->type_ == NodeType::kGraph ||
      node->type_ == NodeType::kOutput) {
    if (compiler_.supported_ops.count(node->operation_)) {
      in_ngraph_ = compiler_.supported_ops[node->operation_](node);
    }
  }
  return in_ngraph_;
}

// Check nodes in NGraph
void Compiler::CheckInNgraph() {
  std::unordered_set<std::string> unsupported_op_names;
  for (const std::shared_ptr<ngraph_bridge::Node>& node : ngraph_.nodes_) {
    node->in_ngraph_ = IsInNGraph(node);
    if (!node->in_ngraph_) {
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
  for (const auto& name : unsupported_op_names) {
    std::cout << "NGRAPH_BRIDGE: Unsupported Op: " << name << std::endl;
  }
}

// Function that parses an nnvm Graph into an intermediary graph
void Compiler::ParseNnvmGraph() {
  // get the shape, data and storage types of all of the nodes
  const auto& idx = graph_.indexed_graph();
  const auto inferred_shapes =
      graph_.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = graph_.GetAttr<std::vector<int>>("dtype");
  const auto& inferred_stypes =
      graph_.GetAttr<mxnet::StorageTypeVector>("storage_type");
  auto get_type = [&idx, &inferred_shapes, &inferred_dtypes, &inferred_stypes](NodePtr node) {
    const uint32_t nid = idx.node_id(node->orig_node_.get());
    const uint32_t eid = idx.entry_id(nid, node->multi_output_index_);
    node->shape_ = inferred_shapes[eid];
    node->dtype_ = inferred_dtypes[eid];
    node->stype_ = inferred_stypes[eid];
  };


  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph_.outputs, [this, &get_type](const nnvm::NodePtr node) {
    const auto& idx = this->graph_.indexed_graph();
    const auto& mutable_nodes = idx.mutable_input_nodes();
    const uint32_t nid = idx.node_id(node.get());
    if (mutable_nodes.count(nid) != 0) {
      // add an auxillary node to the graph
      auto tmpnode = std::make_shared<AuxNode>(node, node->attrs.name);
      get_type(tmpnode);
      this->ngraph_.AddNode(tmpnode);
    } else if (node->is_variable()) {
      // add variable to the graph
      auto tmpnode = std::make_shared<VariableNode>(node, node->attrs.name);
      get_type(tmpnode);
      this->ngraph_.AddNode(tmpnode);
    } else {
      // create operation node
      auto op_name = clean_opname(node->op()->name);
      auto op_node = std::make_shared<OpNode>(node, node->attrs.name, op_name);
      get_type(op_node);
      if (node->num_outputs() > 1) {
        if (IsInNGraph(op_node)) {
          // set up a subgraph for the multi-output op
          auto tmpGraph = std::make_shared<Graph>(node->attrs.name,
                                                  ngraph_.context_, true, node);

          tmpGraph->AddNode(op_node);
          tmpGraph->operation_ = op_name;
          tmpGraph->multi_output_index_ = -1;

          tmpGraph->num_outputs_ = tmpGraph->outputs_.size();
          get_type(tmpGraph);
          this->ngraph_.AddNode(tmpGraph);
          for (size_t i = 0; i < node->num_outputs(); ++i) {
            tmpGraph->outputs_.push_back(op_node);
            auto output = std::make_shared<OutputElement>(tmpGraph, i);
            get_type(output);
            output->name_ = output->name_ + "_" + std::to_string(i);
            output->operation_ = op_name;
            tmpGraph->output_elements_.push_back(output);
            this->ngraph_.AddNode(output);
          }
        } else {
          for (size_t i = 0; i < node->num_outputs(); ++i) {
            auto tmpop = std::make_shared<OpNode>(
                op_node->orig_node_, op_node->name_ + "_" + std::to_string(i),
                op_node->operation_);
            get_type(tmpop);            
            for (auto input : op_node->inputs_) tmpop->inputs_.push_back(input);
            tmpop->multi_output_index_ = i;

            this->ngraph_.AddNode(tmpop);
          }
        }
      } else {
        // add operation
        this->ngraph_.AddNode(op_node);
      }
    }
  });

  // set up the inputs to all of the  nodes in the graph
  for (auto node : ngraph_.nodes_) {
    if (node->type_ != NodeType::kOutput) {
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
  }

  // set up the outputs to the parsed bridge graph
  for (auto e : graph_.outputs) {
    ngraph_.outputs_.push_back(ngraph_[e]);
  }
  // set the shapes on multi-output nodes
  for (auto node : this->ngraph_.nodes_) {
    get_type(node);
    if (node->type_ == NodeType::kGraph) {
      node->shape_ = node->inputs_[0]->shape_;
    }
  }
}

}  // namespace ngraph_bridge
