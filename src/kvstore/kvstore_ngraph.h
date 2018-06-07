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
#ifndef MXNET_KVSTORE_KVSTORE_NGRAPH_H_
#define MXNET_KVSTORE_KVSTORE_NGRAPH_H_

#if MXNET_USE_NGRAPH_DISTRIBUTED

#include <mxnet/kvstore.h>
#include <mpi.h>

namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine using nGraph
 */
class KVStoreNGRAPH : public KVStoreLocal {
 public:
  explicit KVStoreNGRAPH(bool use_device_comm) : KVStoreLocal(use_device_comm) {
      MPI_Init(NULL, NULL);
  }

  virtual ~KVStoreNGRAPH() {
      MPI_Finalize();
  }

  int get_group_size() const override {
      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      return size;
  }

  int get_rank() const override {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank;
  }
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_USE_NGRAPH_DISTRIBUTED
#endif  // MXNET_KVSTORE_KVSTORE_NGRAPH_H_
