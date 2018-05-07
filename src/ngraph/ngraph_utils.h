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
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {

// enable ngraph gluon at runtime.
inline bool ngraph_gluon_enable() {
  return dmlc::GetEnv("MXNET_NGRAPH_GLUON", false);
}

// logging
inline bool ngraph_log_verbose() {
  return dmlc::GetEnv("MXNET_NGRAPH_VERBOSE", false);
}
inline bool ngraph_log_graph() {
  return dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_GRAPH", false);
}
inline bool ngraph_log_viz() {
  return dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_VIZ", false);
}
inline bool ngraph_log_timer() {
  return dmlc::GetEnv("MXNET_NGRAPH_TIMER", false);
}
inline bool ngraph_log_verbose_detail() {
  return dmlc::GetEnv("MXNET_NGRAPH_VERBOSE_DETAIL", false);
}

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

/// Emits a programmer-friendly representation, to assist with logging
/// and debugging.
std::ostream& operator<<(std::ostream& os, const ngraph::Shape& s);

/// Emits a programmer-friendly representation, to assist with logging
/// and debugging.
std::ostream& operator<<(std::ostream& os, const ngraph::AxisSet& s);

/// Emits a programmer-friendly representation, to assist with logging
/// and debugging.
std::ostream& operator<<(std::ostream& os, const nnvm::TShape& s);

/// A convenience method to obtain the elements of s1 that are not
/// present in s2.
template <typename T>
std::set<T> set_subtract(const std::set<T>& s1, const std::set<T>& s2) {
  std::set<T> s3;
  std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                      std::inserter(s3, s3.end()));
  return s3;
}

/// Return the set of axes present in the specified shape.
/// Assume that the shape's axes are numbered consecutively starting at zero.
ngraph::AxisSet shape_to_axis_set(const ngraph::Shape& s);

/// Given the graph node \param n, return the subset of \param n's axes that
/// would
/// remain after removing the axes specified by \param a.
/// Throw an exception if \param a is not a subset of \param n's axes.
///
/// This is useful for inverting the set of reduction axes when calling
/// functions
/// ngraph::builder::mean, etc.
ngraph::AxisSet ngraph_remaining_axes(const NgraphNodePtr& n,
                                      const ngraph::AxisSet& a);

template <typename T>
std::ostream& container_to_debug_stream(
    std::ostream& os, const T& container, const std::string separator = ", ",
    const std::string opening_delimiter = "[",
    const std::string closing_delimiter = "]") {
  os << opening_delimiter;

  bool is_first = true;
  for (const auto& element : container) {
    if (is_first) {
      is_first = false;
    } else {
      os << separator;
    }

    os << element;
  }

  return os;
}

// generates hash for any standard type val, and combines with seed.
template <typename T>
inline std::size_t hash_combine(const std::size_t& seed, const T& val) {
  return seed + std::hash<T>()(val) + (seed << 1);
}

/// Use ngraph::serialize(...) to create a JSON rendition of 'f'.
/// Compute the filename based on this function's parameters.
/// If a file with that name already exists, overwrite it.
void dump_graph(std::shared_ptr<ngraph::Function> f, std::string src_loc = "",
                std::string filename_suffix = "");

// We define the term "vector plus axes" to refer to tensor shapes meeting the
// following criteria:
// For some shape 'S' to be a vector-plus-axes shape:
//   - 'S' has rank >= 1.
//   - At most one axis of 'S' is considered to be the 'vector' axis.  The
//   vector axis may have
//     may have span = 1.
//   - All other axes have span = 1.
//
// These shapes sometimes arise in the processing of image data.  By way of
// example, one typical
// format for image-related tensors is [N,C,H,W].  Some intermediate nodes in
// these graphs produce
// or consume tensors of shape [1,C,1,1].  We coin the term "vector plus axes"
// to describe such
// shapes.

/// Return true iff 's' meets the definition of a 'vector-plus-axes' shape,
/// false if not.
bool has_vector_plus_axes_shape(const ngraph::Shape& s);

/// Create a vector-plus-axes shape with the specified characteristics.
ngraph::Shape get_vector_plus_axes_shape(const size_t rank,
                                         const size_t vector_axis,
                                         const size_t vector_length);

// Assume that 'n' has a vector-plus-axes shape.  Return an operator that (if
// necessary) reshapes
// 'n' to continue to (still) have a vector-plus-axes shape, with rank
// 'output_rank' and with the
// vector-axis specified by 'output_vector_axis'.
//
// If 'n' already meets those criteria, simply return 'n'.
NgraphNodePtr ensure_vector_plus_axes_shape(const NgraphNodePtr n,
                                            const size_t output_rank,
                                            const size_t output_vector_axis);

// Assume that 'n' has vector-plus-axes shape.  Return a node that (if
// necessary) reshapes 'n' to
// remove all axes other than the vector axis.  Examples:
//   [1,C,1,1] --> [C]
//   [1,1,1,1] --> [1]
//   [C]       --> [C]
//   []        --> error // not in vector-plus-axes form
//   [1,C,0,1] --> error // not in vector-plus-axes form
//
// If 'n' already has the required shape, return 'n'.
NgraphNodePtr ensure_vector_only_shape(const NgraphNodePtr n);

}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_UTILS_H_
