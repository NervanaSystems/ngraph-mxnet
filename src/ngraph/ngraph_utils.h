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

#ifndef NGRAPH_UTILS_H_
#define NGRAPH_UTILS_H_

#include <Python.h>
#include "pybind11/pybind11.h"
using namespace pybind11::literals;

namespace ngraph {
namespace py = pybind11;
// Singleton Function to initialize python interpreter
void InitializePython();

// class to lock and release interpreter
class gil_state {
 public:
  gil_state() : m_gstate{PyGILState_Ensure()} {}
  ~gil_state() { PyGILState_Release(m_gstate); }

 private:
  PyGILState_STATE m_gstate;
};

}  // namespace ngraph
#endif  // UTILS_H_
