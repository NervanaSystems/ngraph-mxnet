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

#ifndef NGRAPH_IMPERATIVE_H_
#define NGRAPH_IMPERATIVE_H_

#include <nnvm/op.h>
#include "ngraph_compiler.h"
#include "ngraph_graph.h"

namespace ngraph_bridge {

// Registers ngraph operators with nnvm
void InitImperative();

// Imperative interface for ngraph_bridge
class NGImperative : public Compiler {
 public:
  // NGImperative constructor for mxnet compute kernel(s)
  NGImperative(const nnvm::NodeAttrs &attrs, const mxnet::Context &ctx,
               const std::vector<mxnet::TBlob> &inputs,
               const std::vector<mxnet::OpReqType> *req,
               const std::vector<mxnet::TBlob> &outputs);

  // return ngraph representing the imperative compute kernel
  inline std::shared_ptr<Graph> get_op_ngraph() {
    if (!op_ngraph_) parse_ngraph();
    return op_ngraph_;
  }

  // check for ops supported by ngraph_bridge and imperative interface
  static bool check_op_supported(std::string op_name) {
    static OpEmitter emitter_funcs = Emitter().ngraph_op_funcs_;
    static std::unordered_set<std::string> layer_and_other{
        "split", "SliceChannel", "Activation"};

    if (emitter_funcs.count(op_name) || layer_and_other.count(op_name) ||
        nameswitch.count(op_name))
      return true;

    return false;
  }

 protected:
  std::shared_ptr<Graph> op_ngraph_;
  void parse_ngraph();
};

}  // namespace ngraph_bridge
#endif  // NGRAPH_IMPERATIVE_H_
