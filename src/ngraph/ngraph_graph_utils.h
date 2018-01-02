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

#ifndef MXNET_NGRAPH_NGRAPH_GRAPH_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_GRAPH_UTILS_H_

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace ngraph_bridge {

// Forward Delcaration for type aliases
class Node;
class Graph;
using NodePtr = std::shared_ptr<Node>;
using GraphPtr = std::shared_ptr<Graph>;

// create a random string to avoid subgraph name collisions
inline std::string randomString(const int length = 12) {
  static const char alphabet[] =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789";
  // set up random number generation
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<> dist(
      0, sizeof(alphabet) / sizeof(*alphabet) - 2);
  // create and return string
  std::string str;
  str.reserve(length);
  std::generate_n(std::back_inserter(str), length,
                  [&]() { return alphabet[dist(rng)]; });
  return str;
}

template <typename T>
inline bool in_vec(const std::vector<T>& vec, const T& s) {
  return (std::find(vec.begin(), vec.end(), s) != vec.end());
}

/**
 * Utility for writing a graph to a file for graphviz visualization
 */
void WriteDot(const Graph& graph, const std::string& fname);

/**
 * Write the subgraphs in a graph to a dot file for graphviz visualization
 */
void WriteSubgraphDots(const Graph& graph, const std::string& base);

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_GRAPH_UTILS_H_
