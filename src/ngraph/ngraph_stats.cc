#include "ngraph_stats.h"

#include "ngraph_utils.h"

namespace ngraph_bridge {

std::string exe_mode_to_string(int mode) {
  switch(mode) {
  case 0:
    return "Inference";
  case 1:
    return "Train";
  default:
    return std::to_string(mode);
  }
}
  void NGraphStats::print() {
    if (ngraph_log_timer()) {
      // iterate all the graphs and print their performance stats

      std::vector<ngraph::runtime::PerformanceCounter> forward_perf_;
      std::vector<ngraph::runtime::PerformanceCounter> backward_perf_;
      for (auto& g : graphs_) {
        if (g != nullptr) {
          std::cout << std::string(total_margin_, '#') << "\n";
          std::cout << "# Graph " << g << std::endl;
          auto backend = GetBackendFromContext(g->context_);
          // output inference/training execution mode
          for (int i = 0; i < kGraphExeModeCount; ++i) {
            std::cout << std::string(total_margin_, '=') << "\n";
            std::cout << "- Execution Mode: " << exe_mode_to_string(i) << std::endl;
            {
              std::vector<ngraph::runtime::PerformanceCounter> perf_data =
                  backend->get_performance_data(g->ngraph_forward[i]);
              std::cout << std::string(total_margin_, '-') << "\n";
              std::cout << "- Forward" << std::endl;
              if (perf_data.size() > 0) {
                print_perf_data(perf_data);
                forward_perf_.insert(forward_perf_.end(), perf_data.begin(),
                                     perf_data.end());
              }
            }
            {
              std::vector<ngraph::runtime::PerformanceCounter> perf_data =
                  backend->get_performance_data(g->ngraph_backward[i]);
              std::cout << std::string(total_margin_, '-') << "\n";
              std::cout << "- Backward" << std::endl;
              if (perf_data.size() > 0) {
                print_perf_data(perf_data);
                backward_perf_.insert(backward_perf_.end(), perf_data.begin(),
                                      perf_data.end());
              }
            }
          }
        }
      }

      std::cout << std::string(total_margin_, '#') << "\n";
      std::cout << "# Overall" << std::endl;
      std::cout << std::string(total_margin_, '-') << "\n";
      std::cout << "- Forward" << std::endl;
      print_perf_data(forward_perf_);
      std::cout << std::string(total_margin_, '-') << "\n";
      std::cout << "- Backward" << std::endl;
      print_perf_data(backward_perf_);
      std::cout << std::string(total_margin_, '-') << "\n";
      std::cout << "- Combined" << std::endl;
      std::vector<ngraph::runtime::PerformanceCounter> combined_perf;
      combined_perf.insert(combined_perf.end(), forward_perf_.begin(),
                           forward_perf_.end());
      combined_perf.insert(combined_perf.end(), backward_perf_.begin(),
                           backward_perf_.end());
      print_perf_data(combined_perf);
      std::cout << std::string(total_margin_, '#') << "\n";
    }
  }

  std::multimap<size_t, std::string> NGraphStats::aggregate_timing(
      const std::vector<ngraph::runtime::PerformanceCounter>& perf_data)
  {
    std::unordered_map<std::string, size_t> timing;
    for (const ngraph::runtime::PerformanceCounter& p : perf_data) {
      std::string op = p.name().substr(0, p.name().find('_'));
      timing[op] += p.total_microseconds();
    }

    std::multimap<size_t, std::string> rc;
    for (const std::pair<std::string, size_t>& t : timing) {
      rc.insert({t.second, t.first});
    }
    return rc;
  }

  void NGraphStats::print_perf_data(
      std::vector<ngraph::runtime::PerformanceCounter> perf_data) {
    if (perf_data.size() > 0) {
      std::sort(perf_data.begin(), perf_data.end(),
                [](const ngraph::runtime::PerformanceCounter& p1,
                   const ngraph::runtime::PerformanceCounter& p2) {
                  return p1.total_microseconds() > p2.total_microseconds();
                });
      std::multimap<size_t, std::string> timing = aggregate_timing(perf_data);

      size_t sum = 0;
      std::cout.imbue(std::locale(""));
      for (auto it = timing.rbegin(); it != timing.rend(); it++) {
        std::cout << std::setw(left_margin_) << std::left << it->second
                  << std::setw(right_margin_) << std::right << it->first
                  << "us\n";
        sum += it->first;
      }
      std::cout << std::setw(left_margin_) << std::left << " "
                << std::setw(right_margin_) << std::right
                << std::string(right_margin_ + extra_margin_, '-') << "\n";
      std::cout << std::setw(left_margin_) << std::left
                << "Total:" << std::setw(right_margin_) << std::right << sum
                << "us\n";
      // reset locale
      std::cout.imbue(std::locale::classic());
    }
  }



} // ngraph_bridge
