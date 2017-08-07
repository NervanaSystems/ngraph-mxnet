#include "ngraph_compiler.h"


namespace ngraph{

    PyCompiler::PyCompiler() {
      // init python
      InitializePython();

      // import python modules
      ng_ = py::module::import("ngraph");
      ns_ = py::module::import(
          "ngraph.frontends.tensorflow.tf_importer.ngraph_shaped");
      ngt_ = py::module::import("ngraph.transformers");
      transformer_ = ngt_.attr("make_transformer")();
    }

    // py::object PyCompiler::Compile(Graph g){
    //     gil_state state;
    // }

    void CollapseGraph(const nnvm::Graph& graph){
        auto g = ParseNNVMGraph(graph);
        g.WriteDot("test.dot");
    }

} //end namespace ngraph
