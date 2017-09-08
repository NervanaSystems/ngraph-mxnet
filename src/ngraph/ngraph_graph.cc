#include "ngraph_graph.h"
#include "ngraph_graph_utils.h"
#include "reverse_iterate.h"
#include <functional>
#include <map>
#include <stack>

namespace ngraph {
// Type Aliases
using OpNodePtr = std::shared_ptr<OpNode>;

// Utility for writing a graph to a file for graphviz visualization
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

using edgeRemoveTup = std::tuple<std::string, std::string, bool>;
// Utility for removing bad branches in a directed, acylic subraph.
// Will fail for cyclic graphs
void Graph::RemoveUtil(
    NodePtr s, std::vector<NodePtr>& outNodes,
    std::function<bool(NodePtr)> func,
    std::vector<edgeRemoveTup>& visited_edges) {
  // if this node matches func condition
  if (!func(s))
    outNodes.erase(std::remove(outNodes.begin(), outNodes.end(), s),
                   outNodes.end());
  // visit it's inputs if they're still in the subgraph
  for (auto i : s->inputs)
    if (std::find(outNodes.begin(), outNodes.end(), i) != outNodes.end()) {
      auto edge_tup = edgeRemoveTup{s->name, i->name, func(s)};

      if (std::find(visited_edges.begin(), visited_edges.end(), edge_tup) !=
          visited_edges.end()) {
        visited_edges.push_back(edge_tup);
        RemoveUtil(i, outNodes, func, visited_edges);
      }
    }
}

std::vector<NodePtr> Graph::RemoveBroken(NodePtr s,
                                         std::vector<NodePtr>& subgraph_nodes,
                                         std::function<bool(NodePtr)> func) {
  // if this node matches func condition

  std::vector<NodePtr> outNodes;
  std::function<void(NodePtr)> get_inputs;
  std::vector<edgeRemoveTup > visited_edges;

  get_inputs = [&outNodes, &get_inputs](NodePtr s) {
    if (std::find(outNodes.begin(), outNodes.end(), s) == outNodes.end()) {
      outNodes.emplace_back(s);
    }
    for (auto i : s->inputs)
      if (std::find(outNodes.begin(), outNodes.end(), i) == outNodes.end()) {
        get_inputs(i);
      }
  };
  get_inputs(s);

  bool found_bad = false;
  auto good_subgraph_node = [subgraph_nodes, func,
                             found_bad](NodePtr s) mutable {
    if (!func(s)) found_bad = true;
    if (found_bad) return false;
    if (std::find(subgraph_nodes.begin(), subgraph_nodes.end(), s) !=
        subgraph_nodes.end()) {
      return true;
    } else {
      return false;
    }
  };

  RemoveUtil(s, outNodes, good_subgraph_node, visited_edges);
  return outNodes;
}
// Find a subgraph, check it for bad branches
std::vector<NodePtr> Graph::FindSubgraph(NodePtr s,
                                         std::function<bool(NodePtr)> func) {
  auto subgraph_nodes = DFSselect(s, func);
  std::vector<NodePtr> outNodes;
  if (subgraph_nodes.size() > 2) {
    // search for broken loops
    // remove nodes on broken loops
    outNodes = RemoveBroken(s, subgraph_nodes, func);
  } else {
    outNodes = subgraph_nodes;
  }
  return outNodes;
}

// function to identify and label connected ngraph ops as subgraphs
void Graph::IdentifySubgraphs(std::function<bool(NodePtr)> func) {
  int sg = 1;
  // loop over the nodes from the back
  for (auto i : reverse_iterate(nodes_)) {
    if (i->subgraph == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto subgraph_nodes = FindSubgraph(i, func);
      // if we found a significantly large subgraph, label it
      if (subgraph_nodes.size() > 2) {
        for (auto node : subgraph_nodes)
          if (node->type == NodeType::kOp) node->subgraph = sg;
        sg += 1;
      }
    }
  }
}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void Graph::CollapseSubgraphs() {
  // loop variable for undefined number of subgraphs
  int i = 1;
  while (true) {
    auto tmpGraph = std::make_shared<Graph>();
    // loop over all nodes and add nodes in the current subgraph to
    for (auto node : nodes_)
      if (node->subgraph == i) tmpGraph->AddNode(node);

    if (tmpGraph->nodes_.size() == 0) {
      // if we don't find any nodes, assume we've run out of subgraphs
      break;
    } else {
      // if we found nodes, setup subgraph
      tmpGraph->in_ngraph = true;
      tmpGraph->subgraph = i;
      // set node name and shape based on last node in the subgraph
      auto name = tmpGraph->nodes_.back()->name;
      auto shape = tmpGraph->nodes_.back()->shape;
      tmpGraph->name = "subgraph_" + name + "_" + randomString();
      tmpGraph->shape = shape;
      tmpGraph->dtype = tmpGraph->nodes_.back()->dtype;
      // setup inputs to this subgraph (as a node)
      for (auto node : tmpGraph->nodes_) {
        for (auto input : node->inputs) {
          if (input->subgraph != i) tmpGraph->inputs.emplace_back(input);
        }
      }
      // find the position we're replacing in the graph
      auto it =
          std::find_if(nodes_.begin(), nodes_.end(),
                       [name](NodePtr n) -> bool { return (n->name == name); });
      // insert the subgraph as a node
      nodes_.insert(it, tmpGraph);
      // delete all the ndoes we're replacing with the subgraph
      nodes_.erase(
          std::remove_if(nodes_.begin(), nodes_.end(),
                         [i](NodePtr n) -> bool {
                           return ((n->subgraph == i) &&
                                   (n->type == NodeType::kOp));
                         }),
          nodes_.end());

      // set subgraph as input to all of the nodes downline.
      for (auto n : nodes_)
        for (size_t i = 0; i < n->inputs.size(); ++i)
          if (n->inputs[i]->name == name) n->inputs[i] = tmpGraph;
    }
    i += 1;
  }
}

}  // namespace ngraph