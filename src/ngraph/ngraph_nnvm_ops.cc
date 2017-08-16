#include "ngraph_nnvm_ops.h"
#include <mshadow/tensor.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/symbolic.h>
#include <cstring>
#include "../operator/operator_common.h"
#include "ngraph_utils.h"
#include "pybind11/eval.h"
#include "pybind11/numpy.h"

namespace ngraph {

// get the OP from nnvm, return a pointer to it.
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph) {
  return &(
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(graph->name));
}

void register_forward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto op =
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(graph->name);
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
  // dummy attribute parser for execution
  auto attr_parser = [](nnvm::NodeAttrs* attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      attrs->parsed = std::move(op);
    }
  };
  op.set_attr_parser(attr_parser);

  // register lambda to list inputs
  op.set_attr<nnvm::FListInputNames>(
      "FListInputNames",
      [input_names](const nnvm::NodeAttrs& attrs) { return input_names; });

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
  auto back_op_name = "_backward_" + graph->name;
  op.set_attr<nnvm::FGradient>(
      "FGradient", [back_op_name](const nnvm::NodePtr& n,
                                  const std::vector<nnvm::NodeEntry>& ograds) {
        // auto p = mxnet::op::MakeNode(back_op_name, ,
        // nullptr, &n->attrs.dict, &n);
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

  auto computation = graph->py_computation;
  auto name = graph->name;

  // create the compute lambda
  op.set_attr<mxnet::FCompute>(
      "FCompute<cpu>",
      [computation, name](const nnvm::NodeAttrs& attrs,
                          const mxnet::OpContext& ctx,
                          const std::vector<mxnet::TBlob>& inputs,
                          const std::vector<mxnet::OpReqType>& req,
                          const std::vector<mxnet::TBlob>& outputs) -> void {
        // Lock the gil
        gil_state state;
        // get a tuple of numpy arrays that point to the input data
        py::tuple py_placeholder_vals = py::make_tuple();
        for (size_t i = 0; i < inputs.size(); ++i) {
          // Create py::array of actual value of placeholder[i]
          float* value = (float*)inputs[i].dptr_;
          std::vector<size_t> shape;
          for (size_t j = 0; j < inputs[i].shape_.ndim(); ++j)
            shape.push_back(inputs[i].shape_[j]);
          py::array_t<float> py_placeholder_val(shape, value);
          // push array to placeholder
          py_placeholder_vals = py_placeholder_vals.attr("__add__")(
              py::make_tuple(py_placeholder_val));
        }
        // run the computation
        py::object py_result = (*computation)(*py_placeholder_vals);
        // get the ouput array
        py::array_t<float> py_array_result(py_result);
        void* res_ptr = (void*)py_array_result.request().ptr;
        size_t buffer_size = 4;
        for (size_t i = 0; i < outputs[0].shape_.ndim(); ++i)
          buffer_size *= outputs[0].shape_[i];
        // Memcpy to output
        std::memcpy(outputs[0].dptr_, res_ptr, buffer_size);
      });
}

void register_backward_op(std::shared_ptr<Graph> graph) {
  // register the op with nnvm
  auto op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(
      "_backward_" + graph->name);
  // setup the inputs and outpus
  int num_inputs = graph->inputs.size();
  op.set_num_inputs(num_inputs + 1);
  op.set_num_outputs(num_inputs);
  std::vector<std::string> input_names;

  // dummy attribute parser for execution
  auto attr_parser = [](nnvm::NodeAttrs* attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      attrs->parsed = std::move(op);
    }
  };
  op.set_attr_parser(attr_parser);
  // Mark as backward
  op.set_attr<nnvm::TIsBackward>("TIsBackward", true);

  auto computation = graph->py_backward;
  auto name = graph->name;

  // create the compute lambda
  op.set_attr<mxnet::FCompute>(
      "FCompute<cpu>",
      [computation, name](const nnvm::NodeAttrs& attrs,
                          const mxnet::OpContext& ctx,
                          const std::vector<mxnet::TBlob>& inputs,
                          const std::vector<mxnet::OpReqType>& req,
                          const std::vector<mxnet::TBlob>& outputs) -> void {
        // Lock the gil
        gil_state state;
        // get a tuple of numpy arrays that point to the input data
        py::tuple py_placeholder_vals = py::make_tuple();
        for (size_t i = 0; i < inputs.size(); ++i) {
          // Create py::array of actual value of placeholder[i]
          float* value = (float*)inputs[i].dptr_;
          std::vector<size_t> shape;
          for (size_t j = 0; j < inputs[i].shape_.ndim(); ++j)
            shape.push_back(inputs[i].shape_[j]);
          py::array_t<float> py_placeholder_val(shape, value);
          // push array to placeholder
          py_placeholder_vals = py_placeholder_vals.attr("__add__")(
              py::make_tuple(py_placeholder_val));
        }
        // run the computation
        py::list py_result = (*computation)(*py_placeholder_vals);
        for (size_t j = 0; j < py_result.size(); ++j) {
          // get the ouput array
          py::array_t<float> py_array_result(py_result[j]);
          void* res_ptr = (void*)py_array_result.request().ptr;
          size_t buffer_size = 4;
          for (size_t i = 0; i < outputs[0].shape_.ndim(); ++i)
            buffer_size *= outputs[0].shape_[i];
          // Memcpy to output
          std::memcpy(outputs[j].dptr_, res_ptr, buffer_size);
        }
      });
}
// register subgraph ops with nnvm.
// Breaks if multiple subgraphs (say from)
void register_subgraph(std::shared_ptr<Graph> graph) {
  register_forward_op(graph);
  register_backward_op(graph);
}

}  // namespace ngraph