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
