#include "ngraph_graph.h"
#include <functional>
#include <map>
#include <stack>

namespace ngraph {
// Type Aliases
using OpNodePtr = std::shared_ptr<OpNode>;
using layerGraphs = std::map<std::string, std::function<Graph(const NodePtr)>>;

// Generator to create functions that convert mxnet layer operations
// into a series of ngraph operations
static layerGraphs create_layerGraphs() {
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

// Create dictionary of layer->ngraph functions
auto layer_funcs = create_layerGraphs();

void Graph::WriteDot(const std::string& fname) {
  // open file stream, write graphviz header
  std::ofstream dotfile;
  dotfile.open(fname);
  dotfile << "digraph G { " << std::endl;
  dotfile << "size=\"8,10.5\"" << std::endl;

  // Loop over inputs, write graph connections
  for (auto n : nodes_) {
    for (auto i : n->inputs) {
      dotfile << i->name << " -> " << n->name << ";" << std::endl;
    }
  }
  // Loop over nodes and write labels
  for (auto n : nodes_) {
    dotfile << n->createNodeLabel() << std::endl;
  }
  // Finish file.
  dotfile << "}" << std::endl;
  dotfile.close();
}

// Utility to mark a node as visited and recursive search based on the results
// of an input function
void Graph::DFSUtil(NodePtr s, std::map<std::string, bool>& visited,
                    std::vector<NodePtr>& outNodes,
                    std::function<bool(NodePtr)>& func) {
  // Mark the current node as visited
  visited[s->name] = true;
  // if this node matches func condition
  if (func(s)) {
    // add it to the output
    outNodes.push_back(s);
    // visit it's inputs
    for (auto i : s->inputs) {
      if (!visited[i->name]) {
        DFSUtil(i, visited, outNodes, func);
      }
    }
  }
}

// Depth first selection of nodes based on function criterion
std::vector<NodePtr> Graph::DFSselect(NodePtr s,
                                      std::function<bool(NodePtr)> func) {
  // init visited vector
  std::map<std::string, bool> visited;
  for (auto n : nodes_) visited[n->name] = false;
  // init output vector
  std::vector<NodePtr> outNodes;
  // recursiveliy search the graph
  DFSUtil(s, visited, outNodes, func);
  return outNodes;
}
// Function that parses an nnvm Graph into an intermediary graph
Graph ParseNNVMGraph(nnvm::Graph& graph) {
  // create inermediary graph
  Graph tmpGraph;
  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph.outputs, [&graph, &tmpGraph](const nnvm::NodePtr node) {
    const auto& idx = graph.indexed_graph();

    const auto& mutable_nodes = idx.mutable_input_nodes();
    const uint32_t nid = idx.node_id(node.get());
    if (mutable_nodes.count(nid) != 0){
      // add an auxillary node to the graph
      tmpGraph.AddNode(std::make_shared<AuxNode>(node, node->attrs.name));
    } else if (node->is_variable()) {
      // add variable to the graph
      tmpGraph.AddNode(std::make_shared<VariableNode>(node, node->attrs.name));
    } else {
      // create operation node
      auto op_name = node->op()->name;
      if (op_name.substr(0,9) == "elemwise_"){
        op_name = op_name.substr(9);
      }
      auto op_node =
          std::make_shared<OpNode>(node, node->attrs.name, op_name);
      // setup operation inputs
      for (size_t i = 0; i < node->inputs.size(); ++i) {
          const nnvm::NodeEntry& e = node->inputs[i];
          std::shared_ptr<Node> tmpnode;
          try {
            tmpnode = tmpGraph[e.node->attrs.name];
          } catch (std::string& error) {
            tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
            tmpGraph.AddNode(tmpnode);
          }
          op_node->inputs.emplace_back(tmpnode);
      }
      if (layer_funcs.count(op_node->operation) != 0) {
        // perform layer expansions
        auto tmp = layer_funcs[op_node->operation](op_node);
        for (auto n : tmp.nodes_) 
          tmpGraph.AddNode(n);
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

}  // namespace ngraph