#include "test_util.h"

#include "../../src/ngraph/ngraph_graph.h"
namespace ngraph_bridge {

auto test_node = std::make_shared<nnvm::Node>();
auto test_input = std::make_shared<nnvm::Node>();
std::string test_name = "node_name";
std::vector<NodePtr> test_inputs{
    std::make_shared<VariableNode>(test_input, "test_input")};
std::string test_opname = "relu";

TEST(NGRAPH_GRAPH, VAR_NODE_INIT) {
  EXPECT_EQ(VariableNode(test_node, test_name).type, NodeType::kVariable);
  EXPECT_EQ(VariableNode(test_node, test_name).orig_node, test_node);
  EXPECT_EQ(VariableNode(test_node, test_name).name, test_name);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).type,
            NodeType::kVariable);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).orig_node,
            test_node);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).name, test_name);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).inputs,
            test_inputs);
}

TEST(NGRAPH_GRAPH, AUX_NODE_INIT) {
  EXPECT_EQ(AuxNode(test_node, test_name).type, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name).orig_node, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name).name, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).type, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).orig_node, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).name, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).inputs, test_inputs);
}

TEST(NGRAPH_GRAPH, OP_NODE_INIT) {
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).type, NodeType::kOp);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).orig_node, test_node);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).name, test_name);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).operation, test_opname);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).type,
            NodeType::kOp);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).orig_node,
            test_node);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).name,
            test_name);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).inputs,
            test_inputs);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).operation,
            test_opname);
}

TEST(NGRAPH_GRAPH, GRAPH_INIT) {
  EXPECT_EQ(Graph().type, NodeType::kGraph);
  EXPECT_EQ(Graph().orig_node, nullptr);
  EXPECT_EQ(Graph().name, "");
  EXPECT_EQ(Graph(test_name).type, NodeType::kGraph);
  EXPECT_EQ(Graph(test_name).orig_node, nullptr);
  EXPECT_EQ(Graph(test_name).name, test_name);
}

auto test_ngraph_node = std::make_shared<VariableNode>(test_node, test_name);
struct GraphTest {
  GraphTest() { full_graph.AddNode(test_ngraph_node); }
  Graph empty_graph;
  Graph full_graph;
};
GraphTest test_graphs;

TEST(NGRAPH_GRAPH, GRAPH_NODES_) {
  EXPECT_ANY_THROW(test_graphs.empty_graph["there's no node here"]);
  EXPECT_EQ(test_graphs.full_graph[test_name], test_ngraph_node);
}

auto isop = [](NodePtr s) { return (s->type == NodeType::kOp); };

struct DFS_Test {
  DFS_Test() {
    std::vector<std::string> opnames{"Flatten", "Convolution", "relu", "add",
                                     "FullyConnected"};
    linear_graph.AddNode(std::make_shared<VariableNode>(nullptr, "variable"));
    for (int i = 0; i < 4; ++i)
      linear_graph.AddNode(std::shared_ptr<OpNode>(
          new OpNode(nullptr, "op" + std::to_string(i), opnames[i],
                     {linear_graph.nodes_[i]})));


    branching_graph.AddNode(
        std::make_shared<VariableNode>(nullptr, "variable"));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op0", opnames[0], {branching_graph.nodes_[0]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op1", opnames[1], {branching_graph.nodes_[1]})));
    branching_graph.AddNode(std::shared_ptr<VariableNode>(new VariableNode(
        nullptr, "variable1", {branching_graph.nodes_[1]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op2", opnames[2],
                   {branching_graph.nodes_[2], branching_graph.nodes_[3]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op3", opnames[3],
                   {branching_graph.nodes_[4]})));
    // branching_graph.WriteDot("branching.dot");
  }
  Graph linear_graph;
  Graph branching_graph;
};

DFS_Test test_search;

TEST(NGRAPH_GRAPH, GRAPH_DFS_LINEAR) {
  // TODO
  EXPECT_EQ(test_search.linear_graph
                .DFSselect(test_search.linear_graph.nodes_[4], isop)
                .size(),
            4);
  EXPECT_EQ(test_search.linear_graph
                .DFSselect(test_search.linear_graph.nodes_[3], isop)
                .size(),
            3);
  EXPECT_EQ(test_search.linear_graph
                .DFSselect(test_search.linear_graph.nodes_[0], isop)
                .size(),
            0);
}

TEST(NGRAPH_GRAPH, GRAPH_DFS_BRANCHING) {
  EXPECT_EQ(test_search.branching_graph
                .DFSselect(test_search.branching_graph.nodes_[1], isop)
                .size(),
            1);
  EXPECT_EQ(test_search.branching_graph
                .DFSselect(test_search.branching_graph.nodes_[2], isop)
                .size(),
            2);
  EXPECT_EQ(test_search.branching_graph
                .DFSselect(test_search.branching_graph.nodes_[4], isop)
                .size(),
            3);
  EXPECT_EQ(test_search.branching_graph
                .DFSselect(test_search.branching_graph.nodes_[5], isop)
                .size(),
            4);
}

TEST(NGRAPH_GRAPH, GRAPH_FIND_SUBGRAPH) {
  EXPECT_EQ(test_search.branching_graph
                .FindSubgraph(test_search.branching_graph.nodes_[2], isop)
                .size(),
            2);
  EXPECT_EQ(test_search.branching_graph
                .FindSubgraph(test_search.branching_graph.nodes_[4], isop)
                .size(),
            2);
  EXPECT_EQ(test_search.branching_graph
                .FindSubgraph(test_search.branching_graph.nodes_[5], isop)
                .size(),
            3);
}

TEST(NGRAPH_GRAPH, GRAPH_IDENTIFY_SUBGRAPHS) {
  test_search.branching_graph.IdentifySubgraphs(isop);
  EXPECT_EQ(test_search.branching_graph.nodes_[0]->subgraph, 0);
  EXPECT_EQ(test_search.branching_graph.nodes_[1]->subgraph, -1);
  EXPECT_EQ(test_search.branching_graph.nodes_[2]->subgraph, 1);
  EXPECT_EQ(test_search.branching_graph.nodes_[3]->subgraph, -1);
  EXPECT_EQ(test_search.branching_graph.nodes_[4]->subgraph, 1);
  EXPECT_EQ(test_search.branching_graph.nodes_[5]->subgraph, 1);
}

struct SUBG_test {
  SUBG_test() {
    test.branching_graph.IdentifySubgraphs(isop);
    test.branching_graph.CollapseSubgraphs();
    // test.branching_graph.WriteSubgraphDots("collapsed_branches");
  }
  DFS_Test test;
};

SUBG_test subgraph_test;

TEST(NGRAPH_GRAPH, GRAPH_COLLAPSE_SUBGRAPHS) {
  EXPECT_EQ(subgraph_test.test.branching_graph.nodes_.size(), 4);
  EXPECT_EQ(std::dynamic_pointer_cast<Graph>(
                subgraph_test.test.branching_graph.nodes_.back())
                ->nodes_.size(),
            3);
}

// TEST(NGRAPH_GRAPH, PARSENNVM) {
//   // TODO:: Perhaps in python?
//   EXPECT_EQ(0,1);
// }

}  // namespace ngraph