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

#ifndef MXNET_NGRAPH_NGRAPH_IMPERATIVE_H_
#define MXNET_NGRAPH_NGRAPH_IMPERATIVE_H_

#include <mxnet/ndarray.h>
#include <nnvm/op.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ngraph_compiler.h"
#include "ngraph_graph.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

// Registers ngraph operators with nnvm
void InitImperative();

// Imperative interface for ngraph_bridge
class NGImperative : public Compiler {
 public:
  // NGImperative constructor for mxnet compute kernel(s)
  NGImperative(const nnvm::NodeAttrs &attrs, const mxnet::Context &ctx,
               const std::vector<mxnet::NDArray> &inputs,
               const std::vector<mxnet::OpReqType> *req,
               const std::vector<mxnet::NDArray> &outputs);

  // return ngraph representing the imperative compute kernel
  inline std::shared_ptr<Graph> get_op_ngraph() {
    if (!op_ngraph_) parse_ngraph();
    return op_ngraph_;
  }

  // check for ops supported by ngraph_bridge and imperative interface
  static bool check_op_supported(std::string op_name) {
    static OpEmitter emitter_funcs = Emitter().ngraph_op_funcs_;
    static std::unordered_set<std::string> layer_and_other{"split",
                                                           "SliceChannel"};

    static std::unordered_set<std::string> skip_imperative{
        "expand_dims", "_copy",     "_zeros",
        "zeros_like",  "BatchNorm", "_mul_scalar"};

    if (skip_imperative.count(op_name)) return false;

    if (emitter_funcs.count(op_name) || layer_and_other.count(op_name) ||
        nameswitch.count(op_name))
      return true;

    return false;
  }

 protected:
  std::shared_ptr<Graph> op_ngraph_;
  void parse_ngraph();
};

// op signature: tuple of opname, dev/id, attrs.dict and input dims/types.
using NGIOpKey = std::tuple<const std::string, const std::pair<int, int>,
                            const std::unordered_map<std::string, std::string>,
                            const std::vector<int>>;

// create NGIOpKey for a given NNVM FCompute kernel.
NGIOpKey get_ngiop_key(const nnvm::NodeAttrs &attrs, const mxnet::Context &ctx,
                       const std::vector<mxnet::NDArray> &inputs);

// ngraph cache for imperative ops
// TODO(aemani): potential optimizations w/ LRU, fixed size.
struct NGIOpHash {
  size_t operator()(const NGIOpKey &key) const;
};
struct NGIOpEqual {
  bool operator()(const NGIOpKey &t1, const NGIOpKey &t2) const;
};
using NGIOpCache =
    std::unordered_map<NGIOpKey, std::shared_ptr<Graph>, NGIOpHash, NGIOpEqual>;

}  // namespace ngraph_bridge
#endif  // MXNET_NGRAPH_NGRAPH_IMPERATIVE_H_
