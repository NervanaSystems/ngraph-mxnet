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

#ifndef MXNET_NGRAPH_NGRAPH_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_UTILS_H_
#include <mxnet/ndarray.h>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {

// enable ngraph gluon at runtime.
const bool ngraph_gluon_enable = dmlc::GetEnv("MXNET_NGRAPH_GLUON", false);

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

// This function expects the input string to be of the form
// "(1,2,3)" with optional spaces between the numbers, i.e.
// "(1,2 , 3)". This is the standard format MXNet uses to represent things
// like stride/padding/reshape ordering
template <typename T>
inline std::vector<T> GetIntVectorFromString(std::string input) {
  for (char c : {' ', ')', '(', ']', '['})
    input.erase(std::remove(input.begin(), input.end(), c), input.end());
  std::stringstream ss(input);
  std::vector<T> vect;
  T i;
  while (ss >> i) {
    vect.push_back(i);
    if (ss.peek() == ',') ss.ignore();
  }
  return vect;
}

inline ngraph::AxisVector pyrange(size_t start, size_t stop) {
  ngraph::AxisVector out(stop - start);
  std::iota(out.begin(), out.end(), start);
  return out;
}

inline ngraph::AxisVector pyrange(size_t stop) { return pyrange(0, stop); }

inline std::string get_default(const NodePtr& node, const std::string& key,
                               const std::string default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? node->orig_node_->attrs.dict[key]
             : default_val;
}

inline int get_default(const NodePtr& node, const std::string& key,
                       const int default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? std::stoi(node->orig_node_->attrs.dict[key])
             : default_val;
}

inline float get_default(const NodePtr& node, const std::string& key,
                         const float default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? std::stof(node->orig_node_->attrs.dict[key])
             : default_val;
}

inline bool get_default(const NodePtr& node, const std::string& key,
                        const bool default_val) {
  if (node->orig_node_->attrs.dict.count(key)) {
    const std::string& val = node->orig_node_->attrs.dict[key];
    if (val == "True" || val == "1")
      return true;
    else
      return false;
  }
  return default_val;
}

// check if any ndarray is sparse
inline bool sparse_check(const std::vector<mxnet::NDArray>& ndarray) {
  for (const auto& i : ndarray) {
    if (i.storage_type() != mxnet::kDefaultStorage) return true;
  }
  return false;
}

template <typename T>
inline
    typename std::enable_if<!std::is_unsigned<T>::value, std::vector<T>>::type
    get_default(const NodePtr& node, const std::string& key,
                const std::vector<T>& default_val) {
  return node->orig_node_->attrs.dict.count(key)
             ? GetIntVectorFromString<T>(node->orig_node_->attrs.dict[key])
             : default_val;
}

template <typename T>
inline typename std::enable_if<std::is_unsigned<T>::value, std::vector<T>>::type
get_default(const NodePtr& node, const std::string& key,
            const std::vector<T>& default_val) {
  std::vector<T> out;
  if (node->orig_node_->attrs.dict.count(key)) {
    auto tmp = GetIntVectorFromString<int>(node->orig_node_->attrs.dict[key]);
    for (auto val : tmp) {
      if (val >= 0) {
        out.push_back(val);
      } else {
        throw std::runtime_error(
            std::string("NGRAPH_BRIDGE: expected unsigned integers but got ") +
            std::to_string(val));
      }
    }
  } else {
    out = default_val;
  }
  return out;
}

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_UTILS_H_
