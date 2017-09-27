#include "ngraph_compiler.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_compiler_utils.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph {

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
layerGraphs create_layerGraphs() {
  layerGraphs layer_funcs;
  layer_funcs[std::string("Activation")] = [](const NodePtr node) {
    Graph tmpGraph;
    auto act_type = node->orig_node->attrs.dict["act_type"];
    tmpGraph.AddNode(std::make_shared<OpNode>(node->orig_node, node->name,
                                              act_type, node->inputs));
    return tmpGraph;
  };

  return layer_funcs;
}

// Compiler initialization
Compiler::Compiler(const nnvm::Graph& graph,
                   const nnvm::NodeEntryMap<mxnet::NDArray>& feed_dict){}
  // initialize ngraph_
  DeepCopy(graph);
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
    if (node->type != NodeType::kGraph) {
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
      // setup nnvm node name
      auto name = sg->nodes_.back()->name;

      // use nnvm depth first search to fix node connections in nnvm
      nnvm::DFSVisit(graph_.outputs, [sg_node,
                                      &name](const nnvm::NodePtr node) {
        auto matches_name = [&name](nnvm::NodeEntry n) -> bool {
          return (n.node->attrs.name == name);
        };

        for (auto input : node->inputs) {
          auto it = std::find_if(node->inputs.begin(), node->inputs.end(),
                                 matches_name);

          if (it != node->inputs.end()) {
            node->inputs.insert(it, sg_node);
            node->inputs.erase(std::remove_if(node->inputs.begin(),
                                              node->inputs.end(), matches_name),
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

  return out_graph;
}

nnvm::NodeEntryMap<mxnet::NDArray> Compiler::makeCopiedFeedDict(
    nnvm::NodeEntryMap<mxnet::NDArray> feed_dict) {

  nnvm::NodeEntryMap<mxnet::NDArray> out_dict;
  for (auto kv : feed_dict){
    auto e = kv.first;
    e.node = node_map_[kv.first.node->attrs.name];
    out_dict[e] = kv.second;
  }
  return out_dict;
}

nnvmNodeVec Compiler::GetCopiedNodes(nnvmNodeVec inputs){
  nnvmNodeVec output;
  for (auto node : inputs){
    output.push_back(node_map_[node->attrs.name]);
  }
  return output;
}

nnvm::NodePtr CopyNode(const nnvm::NodePtr& node){
  return std::make_shared<nnvm::Node>(*(node.get()));
}

void Compiler::CopyNodes(const nnvm::Graph& graph) {

  std::function<void(const nnvm::NodePtr&)> copy_nodes;

  copy_nodes = [this, &copy_nodes](const nnvm::NodePtr& node) {

    for (const auto& input : node->inputs) {
      if (!node_map_.count(input.node->attrs.name)) {
        node_map_[input.node->attrs.name] = CopyNode(input.node);
        copy_nodes(node_map_[input.node->attrs.name]);
      }
    }

    for (const auto& input : node->control_deps) {
      if (!node_map_.count(input->attrs.name)) {
        node_map_[input->attrs.name] = CopyNode(input);
        copy_nodes(node_map_[input->attrs.name]);
      }
    }
  };

  for (auto& out : graph.outputs) {
    node_map_[out.node->attrs.name] = CopyNode(out.node);
    copy_nodes(out.node);
  }
}

void Compiler::DeepCopy(nnvm::Graph graph){
  
  CopyNodes(graph);

  std::map<std::string, bool> visited;

  std::function<void(const nnvm::NodePtr&)> set_inputs;

  set_inputs = [&visited, &set_inputs, this](const nnvm::NodePtr& node) {
    int i = 0;
    for (const auto& input : node->inputs) {
      node->inputs[i].node = node_map_[input.node->attrs.name];
      if (!visited.count(input.node->attrs.name)){
        visited[input.node->attrs.name] = true;
        set_inputs(node_map_[input.node->attrs.name]);
      }
      ++i;
    }

    i = 0;
    for (const auto& input : node->control_deps) {
      node->control_deps[i] = node_map_[input->attrs.name];
      if (!visited.count(input->attrs.name)){
        visited[input->attrs.name] = true;
        set_inputs(node_map_[input->attrs.name]);
      }
      ++i;
    }
  };

  graph_.outputs = graph.outputs;
  graph_.attrs = graph.attrs;

  for (auto& out : graph_.outputs) {
    out.node = node_map_[out.node->attrs.name];
    set_inputs(out.node);
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
        for (auto op : compiler_.NgraphOps_) {
          if (node->operation == op) {
            node->in_ngraph = true;
            break;
          }
        }
      }
    } else {
      node->in_ngraph = true;
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
