#ifndef NGRAPH_AUTOBROADCAST_H_
#define NGRAPH_AUTOBROADCAST_H_

#include "ngraph_graph.h"

using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

namespace ngraph_bridge {

class AutoBroadcast {
 private:
  struct Node {
    // pointer to an ngraph node
    // initialized by the constructor
    // conditionally replaced by ReshapeAndBroadcast
    NgraphNodePtr ptr;
    // initial shape of node
    ngraph::Shape shape;
    // shape of node after ngraph::op::Reshape
    ngraph::Shape reshape;
    // axes (0-based) to broadcast by ngraph::op::Broadcast
    ngraph::AxisSet axes;
  } lhs_, rhs_;

  // shape of both nodes after ngraph::op::Broadcast
  ngraph::Shape broadcastshape_;

  // determines whether this class will take any action
  bool RequiresBroadcast();
  
  // set reshape and axes (per node) and broadcast shape
  void SetShapesAndAxes();

  // conditionally replace node with...
  //   ngraph::op::Reshape (if node shape != node reshape) and/or
  //   ngraph::op::Broadcast (if node reshape != broadcast shape)
  //
  // NOTE: Reshape is needed to remove singular dimensions
  //       e.g. when adding (2,3) tensor A to (2,1) tensor B
  //            first Reshape tensor B to (2)
  //            then Broadcast tensor B to (2,3)
  void ReshapeAndBroadcast(Node &node);

 public:
  AutoBroadcast(const NgraphNodePtr &lhsNode, const ngraph::Shape &lhsShape,
                const NgraphNodePtr &rhsNode, const ngraph::Shape &rhsShape);
  NgraphNodePtr lhs() { return lhs_.ptr; }
  NgraphNodePtr rhs() { return rhs_.ptr; }
};

}  // namespace ngraph_bridge

#endif
