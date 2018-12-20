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

#include <nnvm/pass.h>
#include <nnvm/symbolic.h>

#include "test_subgraph_api.h"

TEST_F(SUBGRAPH_API, DUPLICATED_INPUTS) {
  nnvm_graph = nnvm::ApplyPass(std::move(nnvm_graph), "PartitionGraph");
  nnvm::DFSVisit(nnvm_graph.outputs, [](const nnvm::NodePtr node) {
    if (!node->is_variable()) {
      for (const auto& sym : node->attrs.subgraphs) {
        EXPECT_LE(sym->ListInputs(nnvm::Symbol::kAll).size(), 2);
        EXPECT_EQ(sym->outputs.size(), 1);
      }
    }
  });
}
