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
// Utility Function for TBlobs -> vector of ParameterizedTensorView
// TODO: templatize this for non-float vals
inline std::vector<std::shared_ptr<ngraph::runtime::Value > > 
make_placeholder_vals(const std::vector<mxnet::TBlob>& inputs) {
  std::vector<std::shared_ptr<ngraph::runtime::Value > > out;
  for (size_t i = 0; i < inputs.size(); ++i) {
    ngraph::Shape shape;
    for (size_t j = 0; j < inputs[i].shape_.ndim(); ++j) 
      shape.push_back(inputs[i].shape_[j]);
    // TODO: there should be a way to get this froim ParamemeterizeTensorView
    size_t buffer_size = 4; //4 bytes per 32bit float
    for (size_t i = 0; i < inputs[i].shape_.ndim(); ++i)
      buffer_size *= inputs[i].shape_[i];
    auto TV = std::make_shared<
        ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>(
        shape);
    std::memcpy(TV->get_vector().data(), (float*)inputs[i].dptr_, buffer_size);
    out.emplace_back(std::move(TV));
  }
  return out;
}

// TODO: Code copy with make_placeholder_vals, fix?
inline std::vector<std::shared_ptr<ngraph::runtime::Value > > 
make_results(const std::vector<mxnet::TBlob>& outputs) {
  std::vector<std::shared_ptr<ngraph::runtime::Value > > out;
  for (size_t i = 0; i < outputs.size(); ++i) {
    ngraph::Shape shape;
    for (size_t j = 0; j < outputs[i].shape_.ndim(); ++j) 
      shape.push_back(outputs[i].shape_[j]);

    auto TV = std::make_shared<
        ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>(
        shape);
    out.emplace_back(std::move(TV));
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
  size_t buffer_size = 4; //4 bytes per 32bit float
  for (size_t i = 0; i < outputs[outnum].shape_.ndim(); ++i)
    buffer_size *= outputs[outnum].shape_[i];
  // Memcpy to output - TODO::Probably a better way to do this.
  std::memcpy(
      outputs[outnum].dptr_,
      std::dynamic_pointer_cast<
          ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>(
          result)
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
        // get a tuple of numpy arrays that point to the input data
        auto placeholders = make_placeholder_vals(inputs);
        // run the computation
        auto results = make_results(outputs);
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
        // get a tuple of numpy arrays that point to the input data
        auto placeholders = make_placeholder_vals(inputs);
        // run the computation
        auto results = make_results(outputs);
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