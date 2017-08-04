#include <mutex>
#include <cstdlib>

#include "ngraph_utils.h"
#include "Python.h"

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
  PyEval_ReleaseLock();
}

void InitializePython() {
  std::call_once(python_initialized_flag, _InitializePython);
}

}  // namespace ngraph
