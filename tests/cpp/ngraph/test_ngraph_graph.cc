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
  EXPECT_EQ(VariableNode(test_node, test_name).type_, NodeType::kVariable);
  EXPECT_EQ(VariableNode(test_node, test_name).orig_node_, test_node);
  EXPECT_EQ(VariableNode(test_node, test_name).name_, test_name);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).type_,
            NodeType::kVariable);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).orig_node_,
            test_node);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).name_, test_name);
  EXPECT_EQ(VariableNode(test_node, test_name, test_inputs).inputs_,
            test_inputs);
}

TEST(NGRAPH_GRAPH, AUX_NODE_INIT) {
  EXPECT_EQ(AuxNode(test_node, test_name).type_, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name).orig_node_, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name).name_, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).type_, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).orig_node_, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).name_, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).inputs_, test_inputs);
}

TEST(NGRAPH_GRAPH, OP_NODE_INIT) {
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).type_, NodeType::kOp);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).orig_node_, test_node);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).name_, test_name);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname).operation_, test_opname);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).type_,
            NodeType::kOp);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).orig_node_,
            test_node);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).name_,
            test_name);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).inputs_,
            test_inputs);
  EXPECT_EQ(OpNode(test_node, test_name, test_opname, test_inputs).operation_,
            test_opname);
}

TEST(NGRAPH_GRAPH, GRAPH_INIT) {
  EXPECT_EQ(Graph().type_, NodeType::kGraph);
  EXPECT_EQ(Graph().orig_node_, nullptr);
  EXPECT_EQ(Graph(test_name).type_, NodeType::kGraph);
  EXPECT_EQ(Graph(test_name).orig_node_, nullptr);
  EXPECT_EQ(Graph(test_name).name_, test_name);
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

auto isop = [](NodePtr s) { return (s->type_ == NodeType::kOp); };

struct DFS_Test {
  DFS_Test() : linear_graph(), branching_graph() {
    std::vector<std::string> opnames{"Flatten", "Convolution", "relu", "add",
                                     "FullyConnected"};
    linear_graph.AddNode(std::make_shared<VariableNode>(nullptr, "variable"));
    for (int i = 0; i < 4; ++i)
      linear_graph.AddNode(std::shared_ptr<OpNode>(
          new OpNode(nullptr, "op" + std::to_string(i), opnames[i],
                     {linear_graph.GetNodes()[i]})));

    branching_graph.AddNode(
        std::make_shared<VariableNode>(nullptr, "variable"));
    branching_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
        nullptr, "op0", opnames[0], {branching_graph.GetNodes()[0]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
        nullptr, "op1", opnames[1], {branching_graph.GetNodes()[1]})));
    branching_graph.AddNode(std::shared_ptr<VariableNode>(new VariableNode(
        nullptr, "variable1", {branching_graph.GetNodes()[1]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
        nullptr, "op2", opnames[2],
        {branching_graph.GetNodes()[2], branching_graph.GetNodes()[3]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
        nullptr, "op3", opnames[3], {branching_graph.GetNodes()[4]})));
    // branching_graph.WriteDot("branching.dot");
  }
  Graph linear_graph;
  Graph branching_graph;
};

DFS_Test test_search;

TEST(NGRAPH_GRAPH, GRAPH_DFS_LINEAR) {
  // TODO
  EXPECT_EQ(SelectNodes(test_search.linear_graph.GetNodes()[4], isop).size(),
            4);
  EXPECT_EQ(SelectNodes(test_search.linear_graph.GetNodes()[3], isop).size(),
            3);
  EXPECT_EQ(SelectNodes(test_search.linear_graph.GetNodes()[0], isop).size(),
            0);
}

TEST(NGRAPH_GRAPH, GRAPH_DFS_BRANCHING) {
  EXPECT_EQ(SelectNodes(test_search.branching_graph.GetNodes()[1], isop).size(),
            1);
  EXPECT_EQ(SelectNodes(test_search.branching_graph.GetNodes()[2], isop).size(),
            2);
  EXPECT_EQ(SelectNodes(test_search.branching_graph.GetNodes()[4], isop).size(),
            3);
  EXPECT_EQ(SelectNodes(test_search.branching_graph.GetNodes()[5], isop).size(),
            4);
}

TEST(NGRAPH_GRAPH, GRAPH_FIND_SUBGRAPH) {
  EXPECT_EQ(FindSubgraph(test_search.branching_graph,
                         test_search.branching_graph.GetNodes()[2], isop)
                .size(),
            2);
  EXPECT_EQ(FindSubgraph(test_search.branching_graph,
                         test_search.branching_graph.GetNodes()[4], isop)
                .size(),
            2);
  EXPECT_EQ(FindSubgraph(test_search.branching_graph,
                         test_search.branching_graph.GetNodes()[5], isop)
                .size(),
            3);
}

TEST(NGRAPH_GRAPH, GRAPH_IDENTIFY_SUBGRAPHS) {
  IdentifySubgraphs(test_search.branching_graph, isop);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[0]->subgraph_, 0);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[1]->subgraph_, -1);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[2]->subgraph_, 1);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[3]->subgraph_, -1);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[4]->subgraph_, 1);
  EXPECT_EQ(test_search.branching_graph.GetNodes()[5]->subgraph_, 1);
}

struct SUBG_test {
  SUBG_test() {
    IdentifySubgraphs(test.branching_graph, isop);
    CollapseSubgraphs(test.branching_graph);
    // test.branching_graph.WriteSubgraphDots("collapsed_branches");
  }
  DFS_Test test;
};

SUBG_test subgraph_test;

TEST(NGRAPH_GRAPH, GRAPH_COLLAPSE_SUBGRAPHS) {
  EXPECT_EQ(subgraph_test.test.branching_graph.GetNodes().size(), 4);
  EXPECT_EQ(std::dynamic_pointer_cast<Graph>(
                subgraph_test.test.branching_graph.GetNodes().back())
                ->GetNodes()
                .size(),
            3);
}

// TEST(NGRAPH_GRAPH, PARSENNVM) {
//   // TODO:: Perhaps in python?
//   EXPECT_EQ(0,1);
// }

}  // namespace ngraph_bridge
