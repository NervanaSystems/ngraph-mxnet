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

#ifndef NGRAPH_GRAPH_UTILS_H_
#define NGRAPH_GRAPH_UTILS_H_

#include <algorithm>
#include <random>
#include <string>

namespace ngraph_bridge {

class Graph;
using GraphPtr = std::shared_ptr<Graph>;

//create a random string to avoid subgraph name collisions
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

/**
 * Utility for writing a graph to a file for graphviz visualization
 */
void WriteDot(const GraphPtr& graph, const std::string& fname);

/**
 * Write the subgraphs in a graph to a dot file for graphviz visualization
 */
void WriteSubgraphDots(const GraphPtr& graph, const std::string &base);



}  // namespace ngraph
#endif  // UTILS_H_