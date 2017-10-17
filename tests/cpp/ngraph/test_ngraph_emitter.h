#include "test_util.h"
#include "../../src/ngraph/ngraph_emitter.h"

namespace ngraph_bridge{
  
struct testEmitter : public Emitter {
  std::shared_ptr<VariableNode> in1;
  std::shared_ptr<VariableNode> in2;
  std::shared_ptr<OpNode> node;
  NgraphNodePtr data1;
  NgraphNodePtr data2;
  testEmitter() {
      in1 = std::make_shared<VariableNode>(nullptr, "in1");
      in2 = std::make_shared<VariableNode>(nullptr, "in2");
      node = std::make_shared<OpNode>(nullptr, "node", "test",
                                      std::vector<NodePtr>{in1, in2});

      op_map[in1] = std::make_shared<ngraph::op::Parameter>();
      op_map[in2] = std::make_shared<ngraph::op::Parameter>();
      data1 = op_map[in1];
      data2 = op_map[in2];
  };
};

}