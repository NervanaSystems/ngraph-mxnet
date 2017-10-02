#include "ngraph_emitter.h"
#include "ngraph_sgcompiler_utils.h"

namespace ngraph_bridge {

// Compiter initialization
Emitter::Emitter() {
  // Create Operation Maps
  create_UnaryOps();
  create_BinaryOps();
  create_LayerOps();

  // Find all the valid operation names
  for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphLayerOps_) NgraphOps_.emplace_back(x.first);
}


// unary op genrating function generator
void Emitter::create_UnaryOps() {
}

// binary op generating function generator
void Emitter::create_BinaryOps() {
}

// MXNet high level ops generating function
void Emitter::create_LayerOps() {
}
}  // end namespace ngraph
