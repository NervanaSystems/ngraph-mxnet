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

#include "ngraph_compiler.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "../executor/exec_pass.h"
#include "ngraph_nnvm_ops.h"
#include "nnvm/tuple.h"

namespace ngraph_bridge {

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodeEntry CreateNNVMNode(std::shared_ptr<Graph> subgraph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = subgraph->name;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(subgraph);
  // setup the ninputs to the node
  for (auto input : subgraph->inputs)
    node->inputs.emplace_back(nnvm::NodeEntry{input->orig_node, 0, 0});
  // create dummy node parameters
  NGraphParam op;
  node->attrs.parsed = std::move(op);

  // init and return NodeEntry
  return nnvm::NodeEntry{node, 0, 0};
}

// Generator to create functions that convert mxnet layer operations
// into a series of ngraph operations
LayerGraphs create_layerGraphs() {
  LayerGraphs layer_funcs;

  layer_funcs[std::string("Activation")] = [](const NodePtr node) {
    Graph tmpGraph;
    auto act_type = node->orig_node->attrs.dict["act_type"];
    tmpGraph.AddNode(std::make_shared<OpNode>(node->orig_node, node->name,
                                              act_type, node->inputs));
    return tmpGraph;
  };

  return layer_funcs;
}

// infer nnvm::Graph shape and dtype for bind case
// reused from GraphExecutor::Init in graph_executor.cc
void Compiler::Infer(const BindArg* bind) {
  const auto& idx = graph_.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < bind->numForwardInputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      shapes_.push_back(bind->auxStates_[aux_top].shape());
      dtypes_.push_back(bind->auxStates_[aux_top].dtype());
      ++aux_top;
    } else {
      shapes_.push_back(bind->inArgs_[arg_top].shape());
      dtypes_.push_back(bind->inArgs_[arg_top].dtype());
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
  for (size_t i = 0; i < simplebind->numForwardInputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it1 = simplebind->shapeMap_.find(name);
    if (simplebind->shapeMap_.end() != it1) {
      shapes_[i] = it1->second;
    }
    auto it2 = simplebind->dtypeMap_.find(name);
    if (simplebind->dtypeMap_.end() != it2) {
      dtypes_[i] = it2->second;
    }
  }
}

// Compiler initialization
Compiler::Compiler(const nnvm::Graph& graph, const NDArrayMap& feed_dict,
                   const NNVMNodeVec& inputs, const BindArgBase& bindbase) {
  DeepCopy(graph);

  // infer nnvm::Graph shape and type
  auto bind = dynamic_cast<const BindArg*>(&bindbase);
  auto simplebind = dynamic_cast<const SimpleBindArg*>(&bindbase);
  if (bind != nullptr) {
    Infer(bind);
  } else if (simplebind != nullptr) {
    Infer(simplebind);
  }

  graph_ = mxnet::exec::InferShape(std::move(graph_), std::move(shapes_),
                                   "__shape__");
  // TODO: may or may not need error checking
  // if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
  //  HandleInferShapeError(num_forward_inputs, g.indexed_graph(),
  //    g.GetAttr<nnvm::ShapeVector>("shape"));
  //}

  graph_ = mxnet::exec::InferType(std::move(graph_), std::move(dtypes_),
                                  "__dtype__");
  // TODO: may or may not need error checking
  // if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
  //  HandleInferTypeError(num_forward_inputs, g.indexed_graph(),
  //    g.GetAttr<nnvm::DTypeVector>("dtype"));
  //}

  makeCopiedInputs(inputs);
  makeCopiedFeedDict(feed_dict);
  ParseNNVMGraph();
  CheckInNGraph();

  ngraph_.IdentifySubgraphs([&feed_dict](NodePtr s) {
    bool in_feed_dict = false;
    for (auto kv : feed_dict) {
      if (kv.first.node->attrs.name == s->name) {
        in_feed_dict = true;
        break;
      }
    }
    return (s->in_ngraph && s->type == NodeType::kOp && !in_feed_dict);
  });
}

// Main compilation function
nnvm::Graph Compiler::Compile() {
  // Output Graphviz dot files (pre collapse) for vizualization
  if (false) ngraph_.WriteSubgraphDots("pre_collapse");

  ngraph_.CollapseSubgraphs();

  // Output Graphviz dot files (post collapse) for vizualization
  if (false) ngraph_.WriteSubgraphDots("post_collapse");

  for (auto node : ngraph_.nodes_) {
    // store the input variable shape for use by nnvm
    // This is happening because my nnvm graph manipulations are
    // breaking the infer shape functionality, so shapes of inputs
    // don't get properly inferred. Works, because we're inferring
    // the shapes before doing all of this, but not ideal
    if (node->type == NodeType::kAux || node->type == NodeType::kVariable) {
      ngraphShape_[node->name] = node->shape;
      ngraphDtype_[node->name] = node->dtype;
    }
  }

  // find the subgraphs
  for (auto n : ngraph_.nodes_) {
    if (n->type == NodeType::kGraph) {
      // extract and compile subgraph
      auto sg = compiler_.Compile(n);
      // register compiled subgraph with nnvm
      register_subgraph(sg);
      // create nnvm node
      auto sg_node = CreateNNVMNode(sg);

      auto matches = [&sg](nnvm::NodeEntry n) -> bool {
        return (n.node == sg->nodes_.back()->orig_node);
      };

      // Replace outputs if needed
      for (auto& output : graph_.outputs)
        if (matches(output)) output = sg_node;

      // use nnvm depth first search to fix node connections in nnvm
      nnvm::DFSVisit(graph_.outputs, [sg_node,
                                      &matches](const nnvm::NodePtr node) {

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

StateMap Compiler::CopySavedStates(const StateMap& saved_states) {
  StateMap new_saved_states;
  for (auto kv : saved_states) {
    new_saved_states[nodeMap_[kv.first].get()] = kv.second;
  }
  return new_saved_states;
}

void Compiler::makeCopiedFeedDict(const NDArrayMap& feed_dict) {
  for (auto kv : feed_dict) {
    auto e = kv.first;
    e.node = nodeMap_[kv.first.node.get()];
    feedDict_[e] = kv.second;
  }
}

void Compiler::makeCopiedInputs(const NNVMNodeVec& inputs) {
  for (auto node : inputs) {
    inputs_.push_back(nodeMap_[node.get()]);
  }
}

void Compiler::CopyNodes(const nnvm::Graph& graph) {
  // lambda that makes a copy of a node and returns
  // a new smart pointer to that copy
  auto CopyNode = [](const nnvm::NodePtr& node) {
    return std::make_shared<nnvm::Node>(*(node.get()));
  };
  // forward declaration
  std::function<void(const nnvm::NodePtr&)> copy_nodes;

  // function to copy a node and it's inputs based on recursion
  auto copy_and_recurse = [this, &copy_nodes,
                           &CopyNode](const nnvm::NodePtr& node) {
    // check if we've copied this node already
    if (!nodeMap_.count(node.get())) {
      // if we haven't, make and store a copy
      nodeMap_[node.get()] = CopyNode(node);
      // and copy the input nodes
      copy_nodes(nodeMap_[node.get()]);
    }
  };

  // function for copying the inputs of a node
  copy_nodes = [&copy_and_recurse](const nnvm::NodePtr& node) {
    // copy all of the input nodes (and their inputs recursively)
    for (const auto& input : node->inputs) {
      copy_and_recurse(input.node);
    }
    // copy all of the control dependencies
    for (const auto& input : node->control_deps) {
      copy_and_recurse(node);
    }
  };

  // Loop over the output nodes and the nodes and their inputs.
  for (const auto& out : graph.outputs) {
    copy_and_recurse(out.node);
  }
}

void Compiler::DeepCopy(const nnvm::Graph& graph) {
  // make copies of all the graph nodes
  CopyNodes(graph);
  // a map for storing information on where the recursion has visited.
  std::unordered_set<nnvm::NodePtr> visited;

  // forward declare recursive function
  std::function<void(nnvm::NodePtr&)> set_inputs;

  // function to replace a node with a copy and recurse on it's inputs
  auto replace_node_and_recurse = [&visited, &set_inputs,
                                   this](nnvm::NodePtr& node) {
    // check to see if this is an original node or a copied node
    if (nodeMap_.count(node.get())) {
      // if it's original make a copy of the node smart pointer
      nnvm::NodePtr node_copy = node;
      // replace the input node with the copied node
      node = nodeMap_[node_copy.get()];
      // check to see if we've recursed on this node before
      // if we haven't, replace the inputs with copies
      if (!visited.count(node_copy)) {
        visited.insert(node_copy);
        set_inputs(nodeMap_[node_copy.get()]);
      }
    }
  };

  // function to replace the inputs of a node with copies
  set_inputs = [&replace_node_and_recurse](nnvm::NodePtr& node) {
    // replace the input nodes
    for (auto& input : node->inputs) {
      replace_node_and_recurse(input.node);
    }
    // replace the control deps
    for (auto& input : node->control_deps) {
      replace_node_and_recurse(input);
    }
  };

  // init the copied graph
  graph_.outputs = graph.outputs;
  graph_.attrs = graph.attrs;

  // loop over the outputs and replace them
  // not using const references because we're replacing the smart pointer in
  // in the funciton call
  for (auto& out : graph_.outputs) {
    replace_node_and_recurse(out.node);
  }
}

// Check nodes in NGraph
void Compiler::CheckInNGraph() {
  // loop over nodes
  for (auto node : ngraph_.nodes_) {
    if (node->type == NodeType::kOp) {
      // check if it's a multi output related node
      bool multioutput = false;
      auto is_multi = [](NodePtr node) {
        for (auto& kv : node->orig_node->attrs.dict)
          if (kv.first == "num_outputs")
            if (std::stoi(kv.second) > 1) return true;
        return false;
      };
      if (is_multi(node)) multioutput = true;
      for (auto& n : node->inputs)
        if (is_multi(n)) multioutput = true;

      if (multioutput) {
      } else {
        // if it's an operation, check operation name
        if (compiler_.NgraphOpFuncs_.count(node->operation))
          node->in_ngraph = true;
      }
    } else {
      node->in_ngraph = false;
    }
  }
}

// Function that parses an nnvm Graph into an intermediary graph
void Compiler::ParseNNVMGraph() {
  // Create dictionary of layer->ngraph functions
  auto layer_funcs = create_layerGraphs();
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
          tmpnode = this->ngraph_[e.node->attrs.name];
        } catch (std::string& error) {
          try {
            auto name = e.node->attrs.name + "_" + std::to_string(e.index);
            tmpnode = this->ngraph_[name];
          } catch (std::string& error) {
            tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
            this->ngraph_.AddNode(tmpnode);
          }
        }
        op_node->inputs.emplace_back(tmpnode);
      }
      auto expand_layers = [this,
                            &layer_funcs](std::shared_ptr<OpNode>& op_node) {
        auto tmp = layer_funcs[op_node->operation](op_node);
        if (tmp.num_outputs > 1)
          this->ngraph_.nodes_.erase(
              std::remove(this->ngraph_.nodes_.begin(),
                          this->ngraph_.nodes_.end(), op_node),
              this->ngraph_.nodes_.end());

        for (auto n : tmp.nodes_) this->ngraph_.AddNode(n);
      };

      if (layer_funcs.count(op_node->operation) != 0) {
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
    const uint32_t nid = idx.node_id(node->orig_node.get());
    const uint32_t eid = idx.entry_id(nid, 0);
    node->shape = inferred_shapes[eid];
    node->dtype = inferred_dtypes[eid];
  }
}

}  // end namespace ngraph
