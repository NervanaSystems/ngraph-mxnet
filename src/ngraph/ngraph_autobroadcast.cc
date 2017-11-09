
#include <numeric>

#include "ngraph_autobroadcast.h"

namespace ngraph_bridge {

AutoBroadcast::AutoBroadcast(NgraphNodePtr lhsNode,
                             const ngraph::Shape &lhsShape,
                             NgraphNodePtr rhsNode,
                             const ngraph::Shape &rhsShape)
    : lhs_({lhsNode, lhsShape}), rhs_({rhsNode, rhsShape}) {
  if (lhsNode == nullptr || rhsNode == nullptr) {
    throw "NGRAPH_BRIDGE: AutoBroadcast: null input pointer";
  }
  if (lhsShape.size() == 0 || rhsShape.size() == 0) {
    throw "NGRAPH_BRIDGE: AutoBroadcast: empty input shape";
  }
  if (std::find(lhsShape.begin(), lhsShape.end(), 0) != lhsShape.end() ||
      std::find(rhsShape.begin(), rhsShape.end(), 0) != rhsShape.end()) {
    throw "NGRAPH_BRIDGE: AutoBroadcast: invalid input shape";
  }

  // if auto broadcast is necessary
  if (lhs_.shape != rhs_.shape) {
    SetShapesAndAxes();

    // if auto broadcast is possible
    if (broadcastshape_.size()) {
      ReshapeAndBroadcast(lhs_);
      ReshapeAndBroadcast(rhs_);
    }
  }
}  // namespace ngraph_bridge

void AutoBroadcast::SetShapesAndAxes() {
  auto lhsSize = lhs_.shape.size();
  auto rhsSize = rhs_.shape.size();
  auto axis = std::max(lhsSize, rhsSize) - 1;

  // per numpy definition of broadcast:
  // start with trailing dimensions and work forward
  // two dimensions are compatible:
  //  * if they are equal
  //  * if one of them is 1
  while (lhsSize >= 1 || rhsSize >= 1) {
    auto lhsDim = lhsSize ? lhs_.shape[lhsSize - 1] : 1;
    auto rhsDim = rhsSize ? rhs_.shape[rhsSize - 1] : 1;

    if (lhsDim == rhsDim) {
      // add dimension to broadcast shape + lhs/rhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), lhsDim);
      lhs_.reshape.insert(lhs_.reshape.begin(), lhsDim);
      rhs_.reshape.insert(rhs_.reshape.begin(), rhsDim);

    } else if (rhsDim == 1) {
      // add lhs dimension to broadcast shape and lhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), lhsDim);
      lhs_.reshape.insert(lhs_.reshape.begin(), lhsDim);
      // add current axis to rhs broadcast axes
      rhs_.axes.insert(rhs_.axes.begin(), axis);

    } else if (lhsDim == 1) {
      // add rhs dimension to broadcast shape and rhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), rhsDim);
      rhs_.reshape.insert(rhs_.reshape.begin(), rhsDim);
      // add current axis to lhs broadcast axes
      lhs_.axes.insert(lhs_.axes.begin(), axis);

    } else {
      // auto broadcast not possible
      broadcastshape_.clear();
      break;
    }

    if (lhsSize) --lhsSize;
    if (rhsSize) --rhsSize;
    if (axis) --axis;
  }
}

void AutoBroadcast::ReshapeAndBroadcast(Node &node) {
  if (node.shape != node.reshape) {
    // tell reshape to examine input dimensions in order
    ngraph::AxisVector order(node.shape.size());
    std::iota(order.begin(), order.end(), 0);
    node.ptr =
        std::make_shared<ngraph::op::Reshape>(node.ptr, order, node.reshape);
  }

  if (broadcastshape_ != node.reshape) {
    node.ptr = std::make_shared<ngraph::op::Broadcast>(
        node.ptr, broadcastshape_, node.axes);
  }
}

}  // namespace ngraph_bridge
