#include "ngraph_emitter.h"
#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

// Compiter initialization
Emitter::Emitter() {
  // Create Operation Maps
  NgraphUnaryOps_ = create_UnaryOps();
  NgraphBinaryOps_ = create_BinaryOps();
  NgraphLayerOps_ = create_LayerOps();

  // Find all the valid operation names
  for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphLayerOps_) NgraphOps_.emplace_back(x.first);
}


// unary op genrating function generator
UnaryOps Emitter::create_UnaryOps() {
  UnaryOps output;

  return output;
}

// binary op generating function generator
BinaryOps Emitter::create_BinaryOps() {
  BinaryOps output;

  return output;
}

// MXNet high level ops generating function
LayerOps Emitter::create_LayerOps() {
  LayerOps output;

  return output;
}
}  // end namespace ngraph
