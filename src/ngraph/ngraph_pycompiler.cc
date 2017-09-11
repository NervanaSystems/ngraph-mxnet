#include "ngraph_pycompiler.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_pycompiler_utils.h"
#include "ngraph_pyemitter.h"

namespace ngraph {

// Main compilation function
std::shared_ptr<Graph> PyCompiler::Compile(NodePtr graph) {
  // lock gil
  gil_state state;
  // clear the op_map and placeholder_order
  std::map<std::string, py::object> tmp_op_map;
  std::map<std::string, std::vector<std::string> > tmp_placeholder_order;
  op_map = tmp_op_map;
  placeholder_order = tmp_placeholder_order;
  // cast the graph
  auto sg = std::dynamic_pointer_cast<Graph>(graph);
  // compile the subgraph into a python computation
  CompileSubgraph(sg);
  return sg;
}

// Compile a Subgraph into ngraph python objects
void PyCompiler::CompileSubgraph(std::shared_ptr<Graph> graph) {
  // initalize a placeholder order vector for this subgraph
  auto subgraph_name = graph->name;
  std::vector<std::string> tmpvec;
  for (auto i : graph->inputs) tmpvec.push_back(i->name);
  placeholder_order[subgraph_name] = tmpvec;
  // compile the operations
  for (auto node : graph->nodes_) {
    if (NgraphLayerOps_.count(node->operation) == 0) {
      CompileInputs(node, subgraph_name);
    } else if (op_map.count(node->inputs[0]->name) == 0) {
      CompileInput(node->inputs[0], subgraph_name, {});
    }
    CompileNode(node, graph);
    if (false) {
      std::cout << node->name << std::endl;
      for (auto ax : op_map[node->name].attr("axes")) {
        std::cout << ax.attr("name").cast<std::string>() << " "
                  << ax.attr("length").cast<int>() << std::endl;
      }
      std::cout << "-----" << std::endl;
    }
  }

  // create a python tuple of the variable placeholds to compile the computation
  py::tuple py_placeholders = py::make_tuple();
  for (size_t i = 0; i < placeholder_order[subgraph_name].size();
       ++i) {
    auto name = placeholder_order[subgraph_name][int(i)];
    py_placeholders =
        py_placeholders.attr("__add__")(py::make_tuple(op_map[name]));
  }

  auto op = op_map[graph->nodes_.back()->name];
  if (num_axes(op) == 5) {
    auto C = getNthAxis(op, 0);
    auto D = getNthAxis(op, 1);
    auto H = getNthAxis(op, 2);
    auto W = getNthAxis(op, 3);
    auto N = getNthAxis(op, 4);
    // reshape via tensor slice
    // op = slice_tensor(ng, op, pyvec{C, H, W, N});
    op = ng_.attr("tensor_slice")(
        op,
        createPyTuple(pyvec{
            py::slice{0, C.attr("length").cast<int>(), 1}, py::int_{0},
            py::slice{0, H.attr("length").cast<int>(), 1},
            py::slice{0, W.attr("length").cast<int>(), 1},
            py::slice{0, N.attr("length").cast<int>(), 1},
        }),
        createPyTuple(pyvec{C, H, W, N}));
    // reorder axes for mxnet convention.
    op = ng_.attr("axes_with_order")(op, createPyTuple(pyvec{N, C, H, W}));
  }

  // compile the python computation
  graph->py_computation.reset(
      new py::object(transformer_.attr("computation")(op, *py_placeholders)));

  // backward computation
  py::tuple py_deriv_ops = py::make_tuple();
  py::tuple py_shape = createPyTuple(graph->shape);

  auto back_grad =
      ng_.attr("placeholder")(
             op_map[graph->nodes_.back()->name].attr("axes"))
          .attr("named")(graph->name + "_out_grad");

  py::tuple py_back_placeholders = py::make_tuple(back_grad);
  for (size_t i = 0; i < placeholder_order[subgraph_name].size();
       ++i) {
    py_back_placeholders = py_back_placeholders.attr("__add__")(py::make_tuple(
        op_map[placeholder_order[subgraph_name][int(i)]]));
    py_deriv_ops =
        py_deriv_ops.attr("__add__")(py::make_tuple(ng_.attr("deriv")(
            op_map[graph->nodes_.back()->name],
            op_map[placeholder_order[subgraph_name][int(i)]],
            back_grad)));
  }

  // compile the backward computation
  graph->py_backward.reset(new py::object(
      transformer_.attr("computation")(py_deriv_ops, *py_back_placeholders)));
}

// compiling a node
void PyCompiler::CompileNode(NodePtr node, std::shared_ptr<Graph> graph) {
  // if the node has been compiled, return
  if (op_map.count(node->name) > 0) {
    return;
  } else if (NgraphLayerOps_.count(node->operation) != 0) {
    auto data = op_map[node->inputs[0]->name];
    if (num_axes(data) == 4) {
      auto C = getNthAxis(data, 1);
      auto D = make_axis(1, node->name + "_D");
      auto H = getNthAxis(data, 2);
      auto W = getNthAxis(data, 3);
      auto N = getNthAxis(data, 0);
      // reshape the dta
      data = ng_.attr("axes_with_order")(ng_.attr("expand_dims")(data, D, 2),
                                         createPyTuple(pyvec{C, D, H, W, N}));
    } else if (num_axes(data) == 2) {
      auto data_first_axis = getNthAxis(data, 0);
      auto batch_axis =
          getNthAxis(op_map[graph->nodes_[0]->inputs[0]->name], 0);
      if (data_first_axis.attr("length").cast<int>() !=
          batch_axis.attr("length").cast<int>()) {
        data = ng_.attr("Transpose")(data);
      }
      if (data_first_axis != batch_axis) {
        data = ng_.attr("cast_axes")(
            data, createPyTuple(pyvec{batch_axis, getNthAxis(data, 1)}));
      }
    }
    op_map[node->name] =
        NgraphLayerOps_[node->operation](node, data);
  } else if (node->inputs.size() == 1) {
    // get the genrating function for the current operation
    // create the python object for the current operation
    op_map[node->name] = NgraphUnaryOps_[node->operation](
        op_map[node->inputs[0]->name], node->name);
    // compile binary operations, same idea as unary operations
  } else if (node->inputs.size() == 2) {
    op_map[node->name] = NgraphBinaryOps_[node->operation](
        op_map[node->inputs[0]->name],
        op_map[node->inputs[1]->name], node->name);
  } else {
    std::cout << ("operation not yet supported") << std::endl;
    throw;
  }
}

// Compile the inputs, matching ngraph axes.
// This is pretty hacky, primarily for single/double input nodes
// layer ops handled separately
void PyCompiler::CompileInput(NodePtr input, std::string subgraph_name,
                              axes_map node_axes) {
  if (op_map.count(input->name) == 0) {
    axes_map input_axes;
    int axnum = 0;
    for (auto s : input->shape) {
      auto axFound = false;
      for (auto ax : node_axes) {
        if (ax.first.second == s && input_axes.count(ax.first) == 0) {
          axFound = true;
          input_axes[ax.first] = ax.second;
          break;
        }
      }
      if (!axFound) {
        std::ostringstream stream;
        stream << input->name << "_" << axnum;
        auto ax_name = stream.str();
        auto newax = make_axis(s, ax_name);
        input_axes[axes_pair{ax_name, s}] = newax;
      }
      axnum += 1;
    }
    node_axes.insert(input_axes.begin(), input_axes.end());
    pyvec tmp;
    for (auto ax : input_axes) tmp.push_back(ax.second);
    auto axes = createPyTuple(tmp);
    createPyPlaceholder(input->name, axes);
  } else {
    bool first = true;
    for (auto ax : op_map[input->name].attr("axes")) {
      if (first) {
        first = false;
        continue;
      }
      node_axes[axes_pair(ax.attr("name").cast<std::string>(),
                          ax.attr("length").cast<int>())] =
          ax.cast<py::object>();
    }
  }
}

void PyCompiler::CompileInputs(NodePtr node, std::string subgraph_name) {
  axes_map node_axes;
  for (auto input : node->inputs) CompileInput(input, subgraph_name, node_axes);
}

}  // end namespace ngraph
