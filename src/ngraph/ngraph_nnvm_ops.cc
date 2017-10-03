#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/symbolic.h>
#include <cstring>

#include "../operator/operator_common.h"
#include "ngraph_nnvm_ops.h"

namespace ngraph_bridge {


template <typename T>
inline size_t get_buffer_size(const T& shape, size_t nbytes) {
  size_t out = nbytes;
  for (const auto& i : shape)
    out *= i;
  return out;
}

inline ngraph::Shape TShape_to_NShape(const nnvm::TShape& inshape){
  ngraph::Shape shape;
  for (const auto& s : inshape) 
    shape.push_back(s);
  return shape;
}

using ValueVector = std::vector<std::shared_ptr<ngraph::runtime::Value> >;
using ngraph::runtime::ParameterizedTensorView;


template <typename T>
inline std::shared_ptr<ParameterizedTensorView<T> > 
TBlob_to_ParamTensorView(const mxnet::TBlob& input, bool copy = false) {
  auto shape = TShape_to_NShape(input.shape_);
  auto TV = std::make_shared<ParameterizedTensorView<T> >(shape);
  if (copy){
    auto buffer_size = get_buffer_size(shape, 4);
    std::memcpy(TV->get_vector().data(), input.dptr_, buffer_size);
  }
  return TV;
}

inline ValueVector make_ngraph_placeholders(
    const std::vector<mxnet::TBlob>& inputs, bool copy_data = false) {
  ValueVector out;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto TV = TBlob_to_ParamTensorView<ngraph::element::Float32>(inputs[i], copy_data);
    out.emplace_back(TV);
  }
  return out;
}

// Utility function for numpy array outputs -> TBlob
template <typename T>
inline void result_to_TBlob(T& result, 
                            const std::vector<mxnet::TBlob>& outputs,
                            int outnum) {
  // TODO: there should be a way to get this froim ParamemeterizeTensorView through
  // the template
  auto buffer_size = get_buffer_size(outputs[outnum].shape_, 4);
  // Memcpy to output - TODO::Probably a better way to do this.
  std::memcpy(outputs[outnum].dptr_,
              std::dynamic_pointer_cast<
                  ParameterizedTensorView<ngraph::element::Float32>>(result)
                  ->get_vector()
                  .data(),
              buffer_size);
}

// get the OP from nnvm, return a pointer to it.
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph) {
  return &(
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__("ngraph_" + graph->name));
}

void register_forward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto& op =
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__("ngraph_" + graph->name);
  // setup the inputs and outpus
  int num_inputs = graph->inputs.size();
  op.set_num_inputs(num_inputs);
  op.set_num_outputs(1);

  // register the inputs with nnvm
  std::vector<std::string> input_names;
  for (auto n : graph->inputs) {
      input_names.emplace_back(n->name);
      op.add_argument(n->name, "NDArray-or-Symbol", "argument " + n->name);
  }

  // register lambda to list inputs
  op.set_attr<nnvm::FListInputNames>(
      "FListInputNames",
      [input_names](const nnvm::NodeAttrs& attrs) { return input_names; });

  // // get the auxillary inputs
  std::vector<uint32_t> mutate_vars;
  for (size_t i = 0; i < graph->inputs.size(); ++i) {
    if (graph->inputs[i]->type == NodeType::kAux) {
      mutate_vars.emplace_back(i);//graph->inputs[i]->name);
    }
  }

  // register lambda to list inputs
  op.set_attr<nnvm::FMutateInputs>(
      "FMutateInputs",
      [mutate_vars](const nnvm::NodeAttrs& attrs) { return mutate_vars; });

  // dummy attribute parser for execution
  auto attr_parser = [](nnvm::NodeAttrs* attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      attrs->parsed = std::move(op);
    }
  };
  op.set_attr_parser(attr_parser);

  // register lambda to say nothing is inplace
  op.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [num_inputs](const nnvm::NodeAttrs& attrs) {
                                      std::vector<std::pair<int, int>> inplace;
                                      for (int i = 0; i < num_inputs; ++i)
                                        inplace.push_back({i, 0});
                                      return inplace;
                                    });

  // register another lambda to say nothing is in place
  op.set_attr<nnvm::FInplaceIdentity>(
      "FInplaceIdentity", [num_inputs](const nnvm::NodeAttrs& attrs) {
        std::vector<bool> inplace;
        for (int i = 0; i < num_inputs; ++i) inplace.push_back(false);
        return inplace;
      });

  // Register Gradient node generation function
  auto back_op_name = "_backward_" + ("ngraph_" +graph->name);
  op.set_attr<nnvm::FGradient>(
      "FGradient", [back_op_name](const nnvm::NodePtr& n,
                                  const std::vector<nnvm::NodeEntry>& ograds) {
        auto p = nnvm::Node::Create();
        p->attrs.op = nnvm::Op::Get(back_op_name);
        p->attrs.name = n->attrs.name + "_backward";
        p->attrs.dict = n->attrs.dict;
        p->control_deps.emplace_back(n);
        if (p->op()->attr_parser != nullptr) {
          p->op()->attr_parser(&(p->attrs));
        }
        p->inputs.insert(p->inputs.end(), ograds.begin(), ograds.end());
        p->inputs.insert(p->inputs.end(), n->inputs.begin(), n->inputs.end());
        std::vector<nnvm::NodeEntry> ret;
        for (unsigned i = 0; i < p->num_outputs(); ++i) {
          ret.emplace_back(nnvm::NodeEntry{p, i, 0});
        }
        return ret;
      });

  // This is bad. need to redo
  // currently just returing the data types and shapes of the output nodes
  // this subgraph is replacing that were inferred by mxnet
  // not currently checking with the ngraph operations to see if they
  // return the same shape
  auto shape = graph->shape;
  auto dtype = graph->dtype;
  op.set_attr<nnvm::FInferShape>(
      "FInferShape",
      [shape](const nnvm::NodeAttrs& attrs, std::vector<nnvm::TShape>* in_attrs,
              std::vector<nnvm::TShape>* out_attrs) -> bool {
        (*out_attrs)[0] = shape;
        return true;
      });

  // similarly bad
  op.set_attr<nnvm::FInferType>(
      "FInferType",
      [dtype](const nnvm::NodeAttrs& attrs, std::vector<int>* iattr,
              std::vector<int>* oattr) -> bool {
        return mxnet::op::type_assign(&((*oattr)[0]), dtype);
      });

  auto computation = graph->ngraph_forward;
  auto name = graph->name;

  // create the compute lambda
  op.set_attr<mxnet::FCompute>(
      "FCompute<cpu>",
      [computation, name](const nnvm::NodeAttrs& attrs,
                          const mxnet::OpContext& ctx,
                          const std::vector<mxnet::TBlob>& inputs,
                          const std::vector<mxnet::OpReqType>& req,
                          const std::vector<mxnet::TBlob>& outputs) -> void {
        auto cf = computation->make_call_frame();
        auto placeholders = make_ngraph_placeholders(inputs, true);
        auto results = make_ngraph_placeholders(outputs, false);
        (*cf)(placeholders, results); 
        result_to_TBlob(results[0], outputs, 0);
      });
}

void register_backward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto& op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
      "_backward_" + ("ngraph_" + graph->name));
  // setup the inputs and outpus
  int num_inputs = graph->inputs.size();
  op.set_num_inputs(num_inputs + 1);
  op.set_num_outputs(num_inputs);

  // dummy attribute parser for execution
  auto attr_parser = [](nnvm::NodeAttrs* attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      attrs->parsed = std::move(op);
    }
  };
  op.set_attr_parser(attr_parser);
  // Mark as backward
  op.set_attr<bool>("TIsBackward", true);

  auto computation = graph->ngraph_backward;
  auto name = graph->name;
  
  // create the compute lambda
  op.set_attr<mxnet::FCompute>(
      "FCompute<cpu>",
      [computation, name](const nnvm::NodeAttrs& attrs,
                          const mxnet::OpContext& ctx,
                          const std::vector<mxnet::TBlob>& inputs,
                          const std::vector<mxnet::OpReqType>& req,
                          const std::vector<mxnet::TBlob>& outputs) -> void {
        auto cf = computation->make_call_frame();
        auto placeholders = make_ngraph_placeholders(inputs, true);
        auto results = make_ngraph_placeholders(outputs, false);
        (*cf)(placeholders, results);
        for (size_t j = 0; j < results.size(); ++j) 
          result_to_TBlob(results[j], outputs, j);
      });
}
// register subgraph ops with nnvm.
// Breaks if multiple subgraphs (say from)
void register_subgraph(std::shared_ptr<Graph> graph) {
  register_forward_op(graph);
  register_backward_op(graph);
}

}  // namespace ngraph