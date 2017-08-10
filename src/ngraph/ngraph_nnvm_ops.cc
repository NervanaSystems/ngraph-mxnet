#include <mshadow/tensor.h>
#include "ngraph_nnvm_ops.h"
#include "ngraph_utils.h"
#include "pybind11/eval.h"
#include "pybind11/numpy.h"
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <cstring>
#include <mxnet/operator.h>
#include "../operator/operator_common.h"

namespace ngraph {

    void register_subgraph(std::shared_ptr<Graph> graph) {
        auto op = ::dmlc::Registry<::nnvm::Op>::Get(
                        )->__REGISTER_OR_GET__(graph->name);
        op = op.set_num_inputs(graph->inputs.size());                                                
        op = op.set_num_outputs(1)  ;
        std::vector<std::string> input_names;
        for (auto n : graph->inputs) {
            input_names.emplace_back(n->name);                                              
            op.add_argument(n->name, "NDArray-or-Symbol", "argument "+n->name);                  
        }
        op = op.set_attr<nnvm::FListInputNames>("FListInputNames",               
            [input_names](const nnvm::NodeAttrs& attrs) {return input_names;});
        // This is bad. need to redo
        auto shape = graph->shape; 
        auto dtype = graph->dtype;                                                             
        op = op.set_attr<nnvm::FInferShape>("FInferShape", 
            [shape] (const nnvm::NodeAttrs& attrs,
                     std::vector<nnvm::TShape> *in_attrs,
                     std::vector<nnvm::TShape> *out_attrs) -> bool {
                        auto& y = (*out_attrs)[0];
                        auto& x = shape;
                        for (size_t i = 0; i < y.ndim(); ++i) {
                            if (y[i] == 0) {
                                y[i] = x[i];
                            } else if (y[i] != x[i] && x[i] != 0) {
                                return false;
                            }
                        }
                        return true;
                        });  
        //similarly bad
        op = op.set_attr<nnvm::FInferType>("FInferType", 
            [dtype](const nnvm::NodeAttrs& attrs,
                          std::vector<int> *iattr,
                          std::vector<int> *oattr) -> bool{
                return mxnet::op::type_assign(&((*oattr)[0]), dtype);});
        auto computation = graph->py_computation;    
        op.set_attr<mxnet::FCompute>("FCompute<cpu>", 
            [computation](const nnvm::NodeAttrs& attrs,
                          const mxnet::OpContext& ctx,
                          const std::vector<mxnet::TBlob>& inputs,
                          const std::vector<mxnet::OpReqType>& req,
                          const std::vector<mxnet::TBlob>& outputs) -> void {

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

                    py::object py_result =(*computation)(*py_placeholder_vals);
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