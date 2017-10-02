#ifndef NGRAPH_PYCOMPILER_H_
#define NGRAPH_PYCOMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_emitter.h"

namespace ngraph_bridge {

class SGCompiler : public Emitter {
 public:
  SGCompiler(){};
  std::shared_ptr<Graph> Compile(NodePtr graph);
 private:
  // compile subgraph into ngraph python objects
  void CompileSubgraph(std::shared_ptr<Graph> graph);
  // compile inputs to a node
  void CompileInput(NodePtr input);
  void CompileInputs(NodePtr node);
  // compile a single node into an ngraph python object
  void CompileNode(NodePtr node, std::shared_ptr<Graph> graph);
  void ClearOpMap();
};

}  // end namespace ngraph
#endif