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

#include <mxnet/ngraph_context.h>
#include <dmlc/logging.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace ngraph_bridge {

std::unordered_map<std::string, int32_t> SwapMapKeyValue(
    std::unordered_map<int32_t, std::string> input) {
  std::unordered_map<std::string, int32_t> output;
  for (auto kv : input) output.insert({kv.second, kv.first});
  return output;
}

static const std::unordered_map<int32_t, std::string> backends {
  {0, "CPU"}, {1, "IntelGPU"}, {2, "NNP"},
#if MXNET_USE_CUDA
      {3, "GPU"},
#endif
#ifdef MXNET_USE_NGRAPH_IE
      {10, "IE:CPU"}, {11, "IE:GPU"}, {12, "IE:VPU"}, {13, "IE:FPGA"},
#endif
};

static const std::unordered_map<std::string, int32_t> backend_positions =
    SwapMapKeyValue(backends);

int32_t DevIDFromNGraphContext(std::string backend_name, int32_t device_num) {
  int32_t backend_num = 0;
  try {
    backend_num = backend_positions.at(backend_name);
  } catch (...) {
    CHECK(false) << "NGRAPH_BRIDGE: "
                 << "Unsupported backend " << backend_name;
  }
  return device_num + (backend_num << 16);
}

std::pair<std::string, int32_t> NGraphContextFromDevID(int32_t dev_id) {
  int32_t backend_num = dev_id >> 16;
  int32_t device_num = dev_id - (backend_num << 16);
  return {backends.at(backend_num), device_num};
}

}  // namespace ngraph_bridge
