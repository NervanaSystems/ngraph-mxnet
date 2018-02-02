// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

#ifndef MXNET_NGRAPH_NGRAPH_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_UTILS_H_
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace ngraph_bridge {

// enable ngraph gluon at runtime; default enabled.
const bool ngraph_gluon_enable = dmlc::GetEnv("MXNET_NGRAPH_GLUON", true);

// logging
const bool ngraph_log_verbose = dmlc::GetEnv("MXNET_NGRAPH_VERBOSE", false);
const bool ngraph_log_graph = dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_GRAPH", false);
const bool ngraph_log_viz = dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_VIZ", false);
const bool ngraph_log_timer = dmlc::GetEnv("MXNET_NGRAPH_TIMER", false);
const bool ngraph_log_verbose_detail =
    dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_DETAIL", false);

// simple timer for sequential blocks of code
class Timer {
 public:
  // name of timer, print after #printloops.
  static inline void start(std::string name, int printloops = 1) {
    tval e;
    if (tmap().find(name) == tmap().end()) {
      e.sum = e.csum = std::chrono::duration<double>(0);
      e.loops = printloops;
      e.cloops = 0;
    } else {
      e = tmap()[name];
    }
    if (e.loops < 1) return;
    e.start = std::chrono::high_resolution_clock::now();
    tmap()[name] = e;
  }

  // name of timer used in "start"
  static inline void stop(std::string name) {
    if (tmap().find(name) == tmap().end()) return;
    auto d = tmap()[name];
    if (d.loops < 1) return;
    d.cloops++;
    d.csum += (std::chrono::high_resolution_clock::now() - d.start);
    if (d.cloops % d.loops == 0) {
      auto t = d.csum / d.loops;
      d.sum += d.csum;
      d.csum = std::chrono::duration<double>(0);
      std::cout << "NG_TIMER:" << name << ": Current " << t.count() << " Total "
                << d.sum.count() << "ms Iter " << d.cloops << std::endl;
    }
    tmap()[name] = d;
  }

 private:
  struct tval {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::duration<double> csum;
    std::chrono::duration<double> sum;
    size_t loops;
    size_t cloops;
  };

  static inline std::unordered_map<std::string, tval>& tmap() {
    static std::unordered_map<std::string, tval> _tmap;
    return _tmap;
  }
};

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_UTILS_H_
