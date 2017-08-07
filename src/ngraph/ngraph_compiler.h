#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_utils.h"
#include "ngraph_graph.h"
#include "ngraph_emitter.h"

namespace ngraph{

    class PyCompiler{
    public:
        PyCompiler();
        py::object Compile(Graph g);
    private:
        // Nervana Graph imported python module
        py::module np_;
        py::module ng_;
        py::module ns_;
        py::module ngt_;
        py::object transformer_;

    };

    void CollapseGraph(const nnvm::Graph& graph);

} // end namespace ngraph
#endif