#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_utils.h"

namespace ngraph{

    class PyEmitter{
    public:
        PyEmitter::PyEmitter(
        const py::module& ng, const py::module& ns, const py::module& ngt)
        : ng_(ng), ns_(ns), ngt_(ngt){};
    private:
        py::module ng_;
        py::module ns_;
        py::module ngt_; 
    };

}//end namespace ngraph


#endif