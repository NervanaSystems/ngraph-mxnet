#ifndef NGRAPH_EMITTER_H_
#define NGRAPH_EMITTER_H_

#include "ngraph_graph.h"

using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

namespace ngraph_bridge {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject

using OpEmitter =
    std::map<std::string,
             std::function<NgraphNodePtr(const NodePtr&)> >;
// using BinaryOps =
//     std::map<std::string, std::function<NgraphNodePtr(
//                               const NgraphNodePtr&, const NgraphNodePtr&,
//                               const ngraph::element::Type&)> >;
// using LayerOps =
//     std::map<std::string,
//              std::function<NgraphNodePtr(const NodePtr&)> >;

class Emitter {
public:
  Emitter();
  // maps of ngraph operation generator functions
  OpEmitter NgraphOpFuncs_;
protected:
  // create unary operation functions
  void create_UnaryOps();
  // create binary operation functions
  void create_BinaryOps();
  // create larger MXNet layer operations
  void create_LayerOps();

  // information on compiled objects
  std::map<NodePtr, NgraphNodePtr> op_map;
  std::vector<NodePtr> placeholder_order;
};

}  // end namespace ngraph
#endif