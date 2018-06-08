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

#include "ngraph_stats.h"

#include <iomanip>
#include "ngraph_utils.h"

namespace ngraph_bridge {

std::string exe_mode_to_string(int mode) {
  switch (mode) {
    case 0:
      return "Inference";
    case 1:
      return "Train";
    default:
      return std::to_string(mode);
  }
}
void NGraphStats::dump(std::ostream& out) {
  if (ngraph_log_timer()) {
    // accumulator for forward/backward/Combined summary at the end
    const int pass_count = 3;
    const std::string pass_name[pass_count] = {"Forward", "Backward",
                                               "Combined"};
    std::vector<ngraph::runtime::PerformanceCounter> pass_perf[pass_count];

    // iterate all the graphs and print their performance stats
    for (auto& g : graphs_) {
      if (g != nullptr) {
        out << std::string(total_margin_, '#') << "\n";
        out << "# Graph " << g->name_ << std::endl;
        auto backend = GetBackendFromContext(g->context_);

        auto print_perf_for_pass = [&](
            const std::shared_ptr<ngraph::Function>& func, const int pass) {
          std::vector<ngraph::runtime::PerformanceCounter> perf_data =
              backend->get_performance_data(func);
          if (perf_data.size() > 0) {
            out << std::string(total_margin_, '-') << "\n";
            out << "# " + pass_name[pass] << std::endl;
            print_perf_data(out, perf_data);
            pass_perf[pass].insert(pass_perf[pass].end(), perf_data.begin(),
                                   perf_data.end());
          }
        };

        // output inference/training execution mode
        for (int i = 0; i < kGraphExeModeCount; ++i) {
          out << std::string(total_margin_, '=') << "\n";
          out << "# Mode: " << exe_mode_to_string(i) << std::endl;
          print_perf_for_pass(g->ngraph_forward[i], 0);
          print_perf_for_pass(g->ngraph_backward[i], 1);
        }
      }
    }

    out << std::string(total_margin_, '#') << "\n";
    out << "# Overall" << std::endl;
    for (int i = 0; i < pass_count; ++i) {
      out << std::string(total_margin_, '-') << "\n";
      out << "# " + pass_name[i] << std::endl;
      print_perf_data(out, pass_perf[i]);
      // accumulate stats for the last perf counter (total summary)
      if (i != pass_count - 1) {
        pass_perf[pass_count - 1].insert(pass_perf[pass_count - 1].end(),
                                         pass_perf[i].begin(),
                                         pass_perf[i].end());
      }
    }
    out << std::string(total_margin_, '#') << "\n";
  }
}

struct TimeCount {
  size_t time;
  size_t count;
};

struct OpCount {
  std::string op;
  size_t count;
};

std::multimap<size_t, OpCount> aggregate_timing(
    const std::vector<ngraph::runtime::PerformanceCounter>& perf_data) {
  std::unordered_map<std::string, TimeCount> op_timing;
  for (const ngraph::runtime::PerformanceCounter& p : perf_data) {
    std::string op = p.name().substr(0, p.name().find('_'));
    op_timing[op].time += p.total_microseconds();
    op_timing[op].count += 1;
  }

  // multimap will sort by time
  std::multimap<size_t, OpCount> time_op;
  for (const auto& t : op_timing) {
    time_op.insert({t.second.time, {t.first, t.second.count}});
  }
  return time_op;
}

void NGraphStats::print_perf_data(
    std::ostream& out,
    const std::vector<ngraph::runtime::PerformanceCounter>& perf_data) {
  if (perf_data.size() > 0) {
    std::multimap<size_t, OpCount> timing = aggregate_timing(perf_data);

    size_t sum = 0;
    size_t op_count = 0;
    out.imbue(std::locale(""));
    for (auto it = timing.rbegin(); it != timing.rend(); it++) {
      out << std::setw(left_margin_) << std::left
          << it->second.op + " (" + std::to_string(it->second.count) + ")"
          << std::setw(right_margin_) << std::right << it->first << "us\n";
      sum += it->first;
      op_count += it->second.count;
    }
    out << std::setw(left_margin_) << std::left << " "
        << std::setw(right_margin_) << std::right
        << std::string(right_margin_ + extra_margin_, '-') << "\n";
    out << std::setw(left_margin_) << std::left
        << "Total (" + std::to_string(op_count) + "):"
        << std::setw(right_margin_) << std::right << sum << "us\n";
    // reset locale
    out.imbue(std::locale::classic());
  }
}

}  // ngraph_bridge
