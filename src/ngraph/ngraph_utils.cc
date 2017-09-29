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

#include <cstdlib>
#include <mutex>

#include "Python.h"
#include "ngraph_utils.h"

namespace ngraph {

// Flag to indicate whether python has been initialized yet
std::once_flag python_initialized_flag;

void _InitializePython() {
  // This line is necessary to support virtual_envs (along with `--test_env
  // VIRTUAL_ENV` in bazel runner)
  if (auto venv_path = std::getenv("VIRTUAL_ENV")) {
    setenv("PYTHONHOME", venv_path, true);
  }
  Py_Initialize();
  PyEval_InitThreads();
}

void InitializePython() {
  std::call_once(python_initialized_flag, _InitializePython);
}

}  // namespace ngraph
