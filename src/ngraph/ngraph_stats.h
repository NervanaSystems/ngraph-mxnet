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

#ifndef MXNET_NGRAPH_NGRAPH_STATS_H_
#define MXNET_NGRAPH_NGRAPH_STATS_H_

#include <map>
#include <memory>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {

/// A singleton class to track and output nGraph performance statistics.
class NGraphStats {
 public:
  static NGraphStats& get_instance() {
    static NGraphStats instance;
    return instance;
  }

  // disable copy
  NGraphStats(NGraphStats const&) = delete;
  void operator=(NGraphStats const&) = delete;

  void add(const std::shared_ptr<Graph>& g) { graphs_.push_back(g); }
  void dump(std::ostream& stream);

 private:
  // disallow creating instance outside the class
  NGraphStats() {}
  void print_perf_data(
      std::ostream& out,
      const std::vector<ngraph::runtime::PerformanceCounter>& perf_data);

 private:
  std::vector<std::shared_ptr<ngraph_bridge::Graph>> graphs_;
  const int left_column_{40};
  const int right_column_{15};
  const int extra_column_{2};
  const int total_column_{left_column_ + right_column_ + extra_column_};
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_STATS_H_
