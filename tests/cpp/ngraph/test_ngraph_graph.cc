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

#include "test_ngraph_graph.h"
#include "../../src/ngraph/ngraph_graph_utils.h"
#include "../../src/ngraph/ngraph_utils.h"

namespace ngraph_bridge {

TEST_F(NGRAPH_NODE, VAR_NODE_INIT) {
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

TEST_F(NGRAPH_NODE, AUX_NODE_INIT) {
  EXPECT_EQ(AuxNode(test_node, test_name).type_, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name).orig_node_, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name).name_, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).type_, NodeType::kAux);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).orig_node_, test_node);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).name_, test_name);
  EXPECT_EQ(AuxNode(test_node, test_name, test_inputs).inputs_, test_inputs);
}

TEST_F(NGRAPH_NODE, OP_NODE_INIT) {
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

TEST_F(NGRAPH_GRAPH, GRAPH_INIT) {
  EXPECT_EQ(Graph().type_, NodeType::kGraph);
  EXPECT_EQ(Graph().orig_node_, nullptr);
  EXPECT_EQ(Graph(test_name).type_, NodeType::kGraph);
  EXPECT_EQ(Graph(test_name).orig_node_, nullptr);
  EXPECT_EQ(Graph(test_name).name_, test_name);
}

TEST_F(NGRAPH_GRAPH, CYCLIC_GRAPH) {
  auto node = cyclic_graph.nodes_.back();
  EXPECT_ANY_THROW(SelectNodes(node, isop).size());
}

TEST_F(NGRAPH_GRAPH, GRAPH_DFS_LINEAR) {
  EXPECT_EQ(SelectNodes(linear_graph.nodes_[4], isop).size(), 4ul);
  EXPECT_EQ(SelectNodes(linear_graph.nodes_[3], isop).size(), 3ul);
  EXPECT_EQ(SelectNodes(linear_graph.nodes_[0], isop).size(), 0ul);
}

TEST_F(NGRAPH_GRAPH, GRAPH_DFS_BRANCHING) {
  EXPECT_EQ(SelectNodes(branching_graph.nodes_[1], isop).size(), 1ul);
  EXPECT_EQ(SelectNodes(branching_graph.nodes_[2], isop).size(), 2ul);
  EXPECT_EQ(SelectNodes(branching_graph.nodes_[4], isop).size(), 3ul);
  EXPECT_EQ(SelectNodes(branching_graph.nodes_[5], isop).size(), 4ul);
}

TEST_F(NGRAPH_GRAPH, GRAPH_FIND_SUBGRAPH) {
  // branching
  EXPECT_EQ(
      FindSubgraph(branching_graph, branching_graph.nodes_[2], isop).size(),
      2ul);
  EXPECT_EQ(
      FindSubgraph(branching_graph, branching_graph.nodes_[4], isop).size(),
      2ul);
  EXPECT_EQ(
      FindSubgraph(branching_graph, branching_graph.nodes_[5], isop).size(),
      3ul);
  // multi
  EXPECT_EQ(FindSubgraph(multi_graph, multi_graph.nodes_[2], isop).size(), 1ul);
  EXPECT_EQ(FindSubgraph(multi_graph, multi_graph.nodes_[4], isop).size(), 2ul);
  EXPECT_EQ(FindSubgraph(multi_graph, multi_graph.nodes_[5], isop).size(), 2ul);
  EXPECT_EQ(FindSubgraph(multi_graph, multi_graph.nodes_[7], isop).size(), 4ul);
  // complex
  EXPECT_EQ(FindSubgraph(complex_graph, complex_graph.nodes_[9], isop).size(),
            5ul);
  EXPECT_EQ(FindSubgraph(complex_graph, complex_graph.nodes_[20], isop).size(),
            6ul);
  EXPECT_EQ(FindSubgraph(complex_graph, complex_graph.nodes_[24], isop).size(),
            8ul);
  EXPECT_EQ(FindSubgraph(complex_graph, complex_graph.nodes_[25], isop).size(),
            6ul);
}

TEST_F(NGRAPH_GRAPH, GRAPH_COLLAPSE_SUBGRAPHS) {
  IdentifySubgraphs(&branching_graph, isop);
  auto size = branching_graph.nodes_.size();
  EXPECT_EQ(size, 7ul);
  auto subgraph =
      std::dynamic_pointer_cast<Graph>(branching_graph.nodes_[size - 2]);
  EXPECT_NE(subgraph, nullptr);
  EXPECT_EQ(subgraph->nodes_.size(), 3ul);
}

TEST_F(NGRAPH_GRAPH, GRAPH_COLLAPSE_MULTIOUTPUT) {
  if (ngraph_log_viz)
    WriteSubgraphDots(complex_graph, "complex_graph_test_pre_collapse");

  IdentifySubgraphs(&complex_graph, isop);

  if (ngraph_log_viz)
    WriteSubgraphDots(complex_graph, "complex_graph_test_post_collapse");

  int subgraph_count = 0;
  for (auto node : complex_graph.nodes_) {
    if (node->type_ == NodeType::kGraph) {
      subgraph_count += 1;
      auto graph = std::dynamic_pointer_cast<Graph>(node);
      if (graph->subgraph_ == 1) {
        EXPECT_EQ(graph->nodes_.size(), 8ul);
        EXPECT_EQ(graph->outputs_.size(), 3ul);
      } else if (graph->subgraph_ == 2) {
        EXPECT_EQ(graph->nodes_.size(), 4ul);
      }
    }
  }

  EXPECT_EQ(subgraph_count, 6);
}

}  // namespace ngraph_bridge
