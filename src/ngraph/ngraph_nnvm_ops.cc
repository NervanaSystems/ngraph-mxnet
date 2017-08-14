#include <mshadow/tensor.h>
#include "ngraph_nnvm_ops.h"
#include "ngraph_utils.h"
#include "pybind11/eval.h"
#include "pybind11/numpy.h"
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/op_attr_types.h>
#include <cstring>
#include <mxnet/operator.h>
#include "../operator/operator_common.h"



namespace ngraph {
nnvm::Op* get_subgraph_op(std::shared_ptr<Graph> graph) {
  return &(::dmlc::Registry<::nnvm::Op>::Get(
           )->__REGISTER_OR_GET__(graph->name));
}

void register_subgraph(std::shared_ptr<Graph> graph) {
  auto op = ::dmlc::Registry<::nnvm::Op>::Get(
            )->__REGISTER_OR_GET__(graph->name);

  int num_inputs = graph->inputs.size();
  op.set_num_inputs(num_inputs);
  op.set_num_outputs(1);

  std::vector<std::string> input_names;

  for (auto n : graph->inputs) {
    input_names.emplace_back(n->name);
    op.add_argument(n->name, "NDArray-or-Symbol", "argument " + n->name);
  }

  auto attr_parser = [](nnvm::NodeAttrs * attrs) {
    if (attrs->parsed.empty()) {
      NGraphParam op;
      // op.ptr.reset(creator());
      // op.Init(*attrs);
      attrs->parsed = std::move(op);
    }
  };

  op.set_attr_parser(attr_parser);

  op.set_attr<nnvm::FListInputNames>("FListInputNames",
  [input_names](const nnvm::NodeAttrs & attrs) {return input_names;});

  op.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [num_inputs](const nnvm::NodeAttrs & attrs) {
    std::vector<std::pair<int, int> > inplace;
    for (int i = 0; i < num_inputs; ++i)
      inplace.push_back({i , 0});
    return inplace;
  });

  op.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [num_inputs](const nnvm::NodeAttrs & attrs) {
    std::vector<bool> inplace;
    for (int i = 0; i < num_inputs; ++i)
      inplace.push_back(false);
    return inplace;
  });
  
  // This is bad. need to redo
  auto shape = graph->shape;
  auto dtype = graph->dtype;
  op.set_attr<nnvm::FInferShape>(
    "FInferShape",
    [shape] (
      const nnvm::NodeAttrs & attrs,
      std::vector<nnvm::TShape> *in_attrs,
      std::vector<nnvm::TShape> *out_attrs
  ) -> bool {
    (*out_attrs)[0] = shape;
    return true;
  });

  //similarly bad
  op.set_attr<nnvm::FInferType>(
    "FInferType",
    [dtype](
      const nnvm::NodeAttrs & attrs,
      std::vector<int> *iattr,
      std::vector<int> *oattr
    ) -> bool
  {return mxnet::op::type_assign(&((*oattr)[0]), dtype);});


  auto computation = graph->py_computation;
  auto name = graph->name;

  op.set_attr<mxnet::FCompute>(
    "FCompute<cpu>",
    [computation, name](
      const nnvm::NodeAttrs & attrs,
      const mxnet::OpContext & ctx,
      const std::vector<mxnet::TBlob>& inputs,
      const std::vector<mxnet::OpReqType>& req,
      const std::vector<mxnet::TBlob>& outputs
  ) -> void {
    gil_state state;
    py::tuple py_placeholder_vals = py::make_tuple();
    for (size_t i = 0; i < inputs.size(); ++i) {
      // Create py::array of actual value of placeholder[i]
      float* value = (float*) inputs[i].dptr_;
      std::vector<size_t> shape;
      for (size_t j = 0; j < inputs[i].shape_.ndim(); ++j)
        shape.push_back(inputs[i].shape_[j]);
      py::array_t<float> py_placeholder_val(shape, value);

      py_placeholder_vals =
      py_placeholder_vals.attr("__add__")(
        py::make_tuple(py_placeholder_val));
    }

    py::object py_result = (*computation)(*py_placeholder_vals);
    py::array_t<float> py_array_result(py_result);
    void* res_ptr = (void*) py_array_result.request().ptr;
    size_t buffer_size = 4;
    for (size_t i = 0; i < outputs[0].shape_.ndim(); ++i)
      buffer_size *= outputs[0].shape_[i];
    // Memcpy to output
    std::memcpy(outputs[0].dptr_, res_ptr, buffer_size);
  });
}
} // end ngraph namespace