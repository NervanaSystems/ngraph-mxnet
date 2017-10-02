#ifndef NGRAPH_EMITTER_H_
#define NGRAPH_EMITTER_H_

#include "ngraph_graph.h"

using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

namespace ngraph_bridge {
// map aliases for maps of name, function, where function returns an ngraph
// pyobject

using UnaryOps =
    std::map<std::string,
             std::function<NgraphNodePtr(const NgraphNodePtr&, const std::string&)> >;
using BinaryOps =
    std::map<std::string,
             std::function<NgraphNodePtr(const NgraphNodePtr&, const NgraphNodePtr&,
                                      const std::string&)> >;
using LayerOps =
    std::map<std::string,
             std::function<NgraphNodePtr(const NodePtr&, NgraphNodePtr)> >;

class Emitter {
public:
  Emitter();
  // vector of available operations
  std::vector<std::string> NgraphOps_;
protected:
  // create unary operation functions
  UnaryOps create_UnaryOps();
  // create binary operation functions
  BinaryOps create_BinaryOps();
  // create larger MXNet layer operations
  LayerOps create_LayerOps();

  // maps of ngraph operation generator functions
  UnaryOps NgraphUnaryOps_;
  BinaryOps NgraphBinaryOps_;
  LayerOps NgraphLayerOps_;

  // information on compiled objects
  std::map<std::string, NgraphNodePtr> op_map;
  std::vector<std::string> placeholder_order;
};

}  // end namespace ngraph
#endif