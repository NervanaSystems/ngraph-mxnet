#ifndef NGRAPH_COMPILER_H_
#define NGRAPH_COMPILER_H_

#include "ngraph_utils.h"
#include "ngraph_graph.h"
#include "ngraph_emitter.h"

namespace ngraph{
    using UnaryOps = std::map<std::string, std::function<py::object(const py::object&, const std::string&)> >;
    using BinaryOps = std::map<std::string, std::function<py::object(const py::object&, const py::object&, const std::string&)> >;

    class PyCompiler{
    public:
        PyCompiler();
        void Compile(nnvm::Graph& graph, const size_t num_forward_inputs);

    private:
        UnaryOps create_UnaryOps(const py::module& ns, const py::module& ng);
        BinaryOps create_BinaryOps(const py::module& ns, const py::module& ng);
        void CheckGraph(Graph graph);
        void createPyPlaceholder(NodePtr node);
        // void createPyOp(NodePtr node);
        void IdentifySubgraphs(Graph& graph);

        UnaryOps NgraphUnaryOps_;
        BinaryOps NgraphBinaryOps_;
        std::vector<std::string> NgraphOps_;

        py::module np_;
        py::module ng_;
        py::module ns_;
        py::module ngt_;
        py::object transformer_;
        
        std::map<std::string, py::object> op_map;
        std::vector<std::string> placeholder_order;

    };

} // end namespace ngraph
#endif