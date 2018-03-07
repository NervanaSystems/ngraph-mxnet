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

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_

#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>

#include <string>
#include <vector>
#include <iomanip>
#include <iostream>

#include "ngraph_graph.h"

using namespace ngraph;
using namespace std;

static vector<runtime::PerformanceCounter> forward_perf;
static vector<runtime::PerformanceCounter> backward_perf;

inline std::multimap<size_t, std::string>
    aggregate_timing(const std::vector<ngraph::runtime::PerformanceCounter>& perf_data)
{
    std::unordered_map<std::string, size_t> timing;
    for (const ngraph::runtime::PerformanceCounter& p : perf_data)
    {
       std::string op = p.name().substr(0, p.name().find('_'));
       timing[op] += p.total_microseconds();
    }

    std::multimap<size_t, std::string> rc;
    for (const std::pair<std::string, size_t>& t : timing)
    {
       rc.insert({t.second, t.first});
    }
    return rc;
}

inline void print_perf_data(vector<runtime::PerformanceCounter> perf_data) {
  if (perf_data.size() >0) {
      sort(perf_data.begin(),
           perf_data.end(),
           [](const runtime::PerformanceCounter& p1, const runtime::PerformanceCounter& p2) {
              return p1.total_microseconds() > p2.total_microseconds();
           });
      multimap<size_t, string> timing = aggregate_timing(perf_data);
      for (auto it = timing.rbegin(); it != timing.rend(); it++)
      {
         cout.imbue(locale(""));
         cout << setw(15) << left << it->second << " " << setw(10) << right << it->first << "us\n";
      }
  }
}

namespace ngraph_bridge {
// function for returning nnvm::Op corresponding to a subgraph
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph);
// function for registering subgraph operation with nnvm
void register_subgraph(std::shared_ptr<Graph> graph);
// function for computing forward on ngraph
void compute_forward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                     const std::vector<mxnet::NDArray>& inputs,
                     const std::vector<mxnet::OpReqType>& req,
                     const std::vector<mxnet::NDArray>& outputs);
// function for computing backward on ngraph
void compute_backward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                      const std::vector<mxnet::NDArray>& inputs,
                      const std::vector<mxnet::OpReqType>& req,
                      const std::vector<mxnet::NDArray>& outputs);

// dummy parameter struct to match mxnet API
struct NGraphParam {
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  void Init(const nnvm::NodeAttrs& attrs) {}
  // Clean up the graph when this param is deleted
  // if we have 3 or fewer references left
  // forward func/backward func/this param object
  ~NGraphParam() {
    if (g != nullptr && g.use_count() <= 3) {
      std::cout << "cleanup" << std::endl;
      {
          vector<runtime::PerformanceCounter> perf_data = g->ngraph_forward[1]->get_performance_data();
          if (perf_data.size() >0)
              forward_perf.insert(forward_perf.end(), perf_data.begin(), perf_data.end());
          print_perf_data(forward_perf);
      }
      {
          vector<runtime::PerformanceCounter> perf_data = g->ngraph_backward[1]->get_performance_data();
          if (perf_data.size() >0)
              backward_perf.insert(backward_perf.end(), perf_data.begin(), perf_data.end());
          print_perf_data(backward_perf);
      }
      g->CleanUp();
    }
  }
  std::shared_ptr<ngraph_bridge::Graph> g;
};

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
