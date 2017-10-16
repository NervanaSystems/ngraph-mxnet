#ifndef NGRAPH_SGCOMPILER_H_
#define NGRAPH_SGCOMPILER_H_

#include "ngraph_graph.h"
#include "ngraph_emitter.h"

namespace ngraph_bridge {

class SGCompiler : public Emitter {
 public:
  std::shared_ptr<Graph> Compile(NodePtr sub_graph);
 protected:
  // compile subgraph into ngraph python objects
  void CompileSubgraph(std::shared_ptr<Graph> sub_graph);
  // compile input to a node
  void CompileInput(NodePtr input);
  // compile a single node into an ngraph python object
  void CompileNode(NodePtr node, const std::shared_ptr<Graph> sub_graph);
  void ClearOpMap();
};

}  // end namespace ngraph
#endif