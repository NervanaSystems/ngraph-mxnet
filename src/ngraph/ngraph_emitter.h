#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_utils.h"
#include "ngraph_graph.h"

namespace ngraph{

    class PyFactory{
    public:
        PyFactory(
        const py::module& ng, const py::module& ns, const py::module& ngt)
        : ng_(ng), ns_(ns), ngt_(ngt){};

    private:
        py::object CreateVariable(Node n);
        py::object CreateOp(Node n);
        py::module ng_;
        py::module ns_;
        py::module ngt_; 
    };

}//end namespace ngraph


#endif