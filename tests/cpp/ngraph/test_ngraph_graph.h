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
#ifndef TESTS_CPP_NGRAPH_TEST_NGRAPH_GRAPH_H_
#define TESTS_CPP_NGRAPH_TEST_NGRAPH_GRAPH_H_

#include <string>
#include <vector>

#include "test_util.h"

#include "../../src/ngraph/ngraph_graph.h"

namespace ngraph_bridge {

class NGRAPH_NODE : public ::testing::Test {
 protected:
  virtual void SetUp() {
    auto var_node = std::make_shared<VariableNode>(test_input, "test_input");
    test_inputs.push_back(var_node);
  }

  virtual void TearDown() {}

  nnvm::NodePtr test_node;
  nnvm::NodePtr test_input;
  std::string test_name = "node_name";
  std::vector<NodePtr> test_inputs;
  std::string test_opname = "relu";
};

class NGRAPH_GRAPH : public ::testing::Test {
 protected:
  static bool isop(NodePtr s) { return (s->type_ == NodeType::kOp); }

  void CreateLinear() {
    linear_graph.nodes_ = {};
    linear_graph.AddNode(std::make_shared<VariableNode>(nullptr, "variable"));
    for (int i = 0; i < 4; ++i)
      linear_graph.AddNode(std::shared_ptr<OpNode>(
          new OpNode(nullptr, "op" + std::to_string(i), opnames[i],
                     {linear_graph.nodes_[i]})));
  }

  void CreateCyclic() {
    cyclic_graph.nodes_ = {};
    cyclic_graph.AddNode(std::make_shared<VariableNode>(nullptr, "variable"));
    for (int i = 0; i < 4; ++i)
      cyclic_graph.AddNode(std::shared_ptr<OpNode>(
          new OpNode(nullptr, "op" + std::to_string(i), opnames[i],
                     {cyclic_graph.nodes_[i]})));
    cyclic_graph.nodes_[2]->inputs_.push_back(cyclic_graph.nodes_[4]);
  }

  void CreateBranching() {
    branching_graph.nodes_ = {};
    branching_graph.AddNode(
        std::make_shared<VariableNode>(nullptr, "variable"));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op0", opnames[0], {branching_graph.nodes_[0]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op1", opnames[1], {branching_graph.nodes_[1]})));
    branching_graph.AddNode(std::shared_ptr<VariableNode>(
        new VariableNode(nullptr, "variable1", {branching_graph.nodes_[1]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op2", opnames[2],
                   {branching_graph.nodes_[2], branching_graph.nodes_[3]})));
    branching_graph.AddNode(std::shared_ptr<OpNode>(
        new OpNode(nullptr, "op3", opnames[3], {branching_graph.nodes_[4]})));
    branching_graph.AddNode(std::shared_ptr<VariableNode>(
        new VariableNode(nullptr, "variable2", {branching_graph.nodes_[5]})));
  }

  void CreateMultiOut() {
    multi_graph.nodes_ = {};
    std::vector<bool> is_op{0, 0, 1, 0, 1, 1, 0, 1, 0};
    std::vector<std::vector<int>> inputs = {{},  {0},    {0},    {1},   {1, 2},
                                            {2}, {3, 4}, {4, 5}, {6, 7}};

    for (int i = 0; i < 9; ++i) {
      std::vector<NodePtr> input_nodes;
      for (auto n : inputs[i]) input_nodes.push_back(multi_graph.nodes_[n]);

      if (is_op[i]) {
        multi_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
            nullptr, "op" + std::to_string(i), "tanh", input_nodes)));
      } else {
        multi_graph.AddNode(std::make_shared<VariableNode>(
            nullptr, "variable" + std::to_string(i), input_nodes));
      }
    }
  }

  void CreateComplexGraph() {
    complex_graph.nodes_ = {};
    std::vector<bool> is_op(26, true);
    for (auto i : {6, 7, 8, 10, 11, 13, 16, 23}) is_op[i] = false;

    std::vector<std::vector<int>> inputs = {
        {},           {},          {0},      {0},      {1},      {1},
        {},           {2},         {2, 3},   {3, 4},   {4},      {4, 5},
        {5, 6},       {7},         {8},      {9},      {10},     {11},
        {12},         {13, 14},    {14, 15}, {16, 17}, {17, 18}, {},
        {19, 20, 21}, {21, 22, 23}};

    for (size_t i = 0; i < is_op.size(); ++i) {
      std::vector<NodePtr> input_nodes;
      for (auto n : inputs[i]) input_nodes.push_back(complex_graph.nodes_[n]);

      if (is_op[i]) {
        complex_graph.AddNode(std::shared_ptr<OpNode>(new OpNode(
            nullptr, "op" + std::to_string(i), "tanh", input_nodes)));
      } else {
        complex_graph.AddNode(std::make_shared<VariableNode>(
            nullptr, "variable" + std::to_string(i), input_nodes));
      }
    }
  }

  virtual void SetUp() {
    test_ngraph_node = std::make_shared<VariableNode>(test_node, test_name);

    full_graph.AddNode(test_ngraph_node);

    CreateLinear();
    CreateCyclic();
    CreateBranching();
    CreateMultiOut();
    CreateComplexGraph();
  }

  virtual void TearDown() {}

  nnvm::NodePtr test_node;

  std::string test_name = "node_name";
  std::vector<std::string> opnames{"Flatten", "Convolution", "relu", "add",
                                   "FullyConnected"};
  NodePtr test_ngraph_node;
  Graph empty_graph;
  Graph full_graph;
  Graph linear_graph;
  Graph cyclic_graph;
  Graph branching_graph;
  Graph multi_graph;
  Graph complex_graph;
};

}  // namespace ngraph_bridge

#endif  // TESTS_CPP_NGRAPH_TEST_NGRAPH_GRAPH_H_
