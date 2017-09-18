#include "ngraph_compiler.h"
#include "ngraph_compiler_utils.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_nnvm_ops.h"

namespace ngraph {

// Compiter initialization
Compiler::Compiler() {

  NgraphOps_ = compiler_.NgraphOps_;
}

// Main compilation function
nnvm::Graph Compiler::Compile(
    nnvm::Graph graph,
    std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
    std::unordered_map<std::string, int>& arg_dtype_map) {

  auto g = ParseNNVMGraph(graph);

  CheckInNGraph(g);

  g.IdentifySubgraphs(
      [](NodePtr s) { 
        return (s->in_ngraph && s->type == NodeType::kOp); 
      });

  // g.WriteSubgraphDots("pre_collapse");
  g.CollapseSubgraphs();

  // Output Graphviz dot files for vizualization
  if (false) {
    g.WriteSubgraphDots("test");
  }
  // throw;
  for (auto node : g.nodes_) {
    // store the input variable shape for use by nnvm
    // This is happening because my nnvm graph manipulations are
    // breaking the infer shape functionality, so shapes of inputs
    // don't get properly inferred. Works, because we're inferring
    // the shapes before doing all of this, but not ideal
    if (node->type != NodeType::kGraph) {
      arg_shape_map[node->name] = node->shape;
      arg_dtype_map[node->name] = node->dtype;
    }
  }

  // find the subgraphs
  for (auto n : g.nodes_) {
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
      nnvm::DFSVisit(graph.outputs, [sg_node, &name](const nnvm::NodePtr node) {
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
  out_graph.outputs = graph.outputs;

  return out_graph;
}

// Check nodes in NGraph
void Compiler::CheckInNGraph(Graph& graph) {
  // loop over nodes
  for (auto node : graph.nodes_) {
    if (node->type == NodeType::kOp) {
      //check if it's a multi output related node
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
        for (auto op : NgraphOps_) {
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

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodeEntry Compiler::CreateNNVMNode(std::shared_ptr<Graph> graph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = graph->name;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(graph);
  // setup the ninputs to the node
  for (auto input : graph->inputs)
    node->inputs.emplace_back(nnvm::NodeEntry{input->orig_node, 0, 0});
  // create dummy node parameters
  NGraphParam op;
  node->attrs.parsed = std::move(op);

  // init and return NodeEntry
  return nnvm::NodeEntry{node, 0, 0};
}


// Generator to create functions that convert mxnet layer operations
// into a series of ngraph operations
layerGraphs Compiler::create_layerGraphs() {
  layerGraphs layer_funcs;
  layer_funcs[std::string("Activation")] = [](const NodePtr node) {
    Graph tmpGraph;
    auto act_type = node->orig_node->attrs.dict["act_type"];
    tmpGraph.AddNode(std::make_shared<OpNode>(node->orig_node, node->name,
                                              act_type, node->inputs));
    return tmpGraph;
  };

  // layer_funcs[std::string("split")] = [](const NodePtr node) {
  //   Graph tmpGraph(node->name);
  //   int num_outputs = 1;
  //   for (auto& kv : node->orig_node->attrs.dict) 
  //     if (kv.first == "num_outputs") num_outputs = std::stoi(kv.second);
    
  //   tmpGraph.num_outputs = num_outputs;
  //   for (int i = 0; i < num_outputs; ++i) {
  //     auto tmpslice = std::make_shared<OpNode>(
  //         node->orig_node, node->name + "_" + std::to_string(i), "split");
  //     tmpslice->inputs.push_back(node->inputs[0]);
  //     tmpslice->multioutput_index = i;
  //     tmpGraph.AddNode(tmpslice);
  //   }
  //   // TODO: Jayaram says marking the last N-1 slices as inputs to the
  //   // first slice will help transformer optimizations
  //   return tmpGraph;
  // };

  // layer_funcs[std::string("SliceChannel")] = [&layer_funcs](const NodePtr node) {
  //   return layer_funcs["split"](node);
  // };
  return layer_funcs;
}

// Function that parses an nnvm Graph into an intermediary graph
Graph Compiler::ParseNNVMGraph(nnvm::Graph& graph) {
  // create inermediary graph
  Graph tmpGraph;
  // Create dictionary of layer->ngraph functions
  auto layer_funcs = create_layerGraphs();
  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph.outputs, [&graph, &tmpGraph,
                                 &layer_funcs](const nnvm::NodePtr node) {
    const auto& idx = graph.indexed_graph();

    const auto& mutable_nodes = idx.mutable_input_nodes();
    const uint32_t nid = idx.node_id(node.get());
    if (mutable_nodes.count(nid) != 0) {
      // add an auxillary node to the graph
      tmpGraph.AddNode(std::make_shared<AuxNode>(node, node->attrs.name));
    } else if (node->is_variable()) {
      // add variable to the graph
      tmpGraph.AddNode(std::make_shared<VariableNode>(node, node->attrs.name));
    } else {
      // create operation node
      auto op_name = clean_opname(node->op()->name);
      auto op_node = std::make_shared<OpNode>(node, node->attrs.name, op_name);
      // setup operation inputs
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = node->inputs[i];
        std::shared_ptr<Node> tmpnode;
        try {
          tmpnode = tmpGraph[e.node->attrs.name];
        } catch (std::string& error) {
          try {
            auto name = e.node->attrs.name + "_" + std::to_string(e.index);
            tmpnode = tmpGraph[name];
          } catch (std::string& error) {
            tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
            tmpGraph.AddNode(tmpnode);
          }
        }
        op_node->inputs.emplace_back(tmpnode);
      }
      auto expand_layers = [&layer_funcs](std::shared_ptr<OpNode>& op_node,
                                          Graph& tmpGraph) {
        auto tmp = layer_funcs[op_node->operation](op_node);
        if (tmp.num_outputs > 1)
          tmpGraph.nodes_.erase(std::remove(tmpGraph.nodes_.begin(),
                                            tmpGraph.nodes_.end(), op_node),
                                tmpGraph.nodes_.end());

        for (auto n : tmp.nodes_) tmpGraph.AddNode(n);
      };

      if (layer_funcs.count(op_node->operation) != 0) {
        // perform layer expansions
        expand_layers(op_node, tmpGraph);
      } else {
        // add operation
        tmpGraph.AddNode(op_node);
      }
    }
  });

  // get the shape and data types of all of the nodes
  const auto& idx = graph.indexed_graph();
  const auto inferred_shapes =
      graph.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = graph.GetAttr<std::vector<int>>("dtype");
  for (auto node : tmpGraph.nodes_) {
    const uint32_t nid = idx.node_id(node->orig_node.get());
    const uint32_t eid = idx.entry_id(nid, 0);
    node->shape = inferred_shapes[eid];
    node->dtype = inferred_dtypes[eid];
  }
  // return intermediary graph
  return tmpGraph;
}

}  // end namespace ngraph
