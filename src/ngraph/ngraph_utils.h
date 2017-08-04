#ifndef NGRAPH_UTILS_H_
#define NGRAPH_UTILS_H_

#include <Python.h>
#include "pybind11/pybind11.h"

namespace ngraph {
    namespace py = pybind11;
// Runs python initializer, needs to be called ONLY once
    void InitializePython();

    class gil_state {
     public:
      gil_state() : m_gstate{PyGILState_Ensure()} {}
      ~gil_state() { PyGILState_Release(m_gstate); }

     private:
      PyGILState_STATE m_gstate;
    };

}  // namespace ngraph
#endif  // UTILS_H_
