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

#ifndef NGRAPH_COMPILER_UTILS_H_
#define NGRAPH_COMPILER_UTILS_H_

#include <string>

namespace ngraph {

// function to remove modifiers frop op names
inline std::string clean_opname(std::string name) {
  for (std::string str : {"elemwise_", "broadcast_"})
    if (name.substr(0, str.size()) == str) name = name.substr(str.size());
  if (name == "_mul") name = "multiply";
  return name;
}

}  // namespace ngraph
#endif  // UTILS_H_