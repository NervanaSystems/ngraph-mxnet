#include "ngraph_pyemitter.h"
#include "ngraph_pycompiler_utils.h"

namespace ngraph {

// Compiter initialization
PyEmitter::PyEmitter() {
  // init python
  InitializePython();
  gil_state state;

  // import python modules
  ng_ = py::module::import("ngraph");
  ngt_ = py::module::import("ngraph.transformers");
  transformer_ = ngt_.attr("make_transformer")();

  // Create Operation Maps
  NgraphUnaryOps_ = create_UnaryOps(ng_);
  NgraphBinaryOps_ = create_BinaryOps(ng_);
  NgraphLayerOps_ = create_LayerOps(ng_);

  // Find all the valid operation names
  for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphLayerOps_) NgraphOps_.emplace_back(x.first);
}

// Utility for creating a named ngraph axis
py::object PyEmitter::make_axis(int length, std::string name) {
  return ng_.attr("make_axis")("length"_a = length, "name"_a = name);
}

// Function for emitting a variable placeholder in ngraph
py::object PyEmitter::createPyPlaceholder(std::string name, py::tuple axes) {
  // create placeholder in python, store it in class dictionary
  auto op = ng_.attr("placeholder")(axes).attr("named")(name);
  op_map[name] = op;
  return op;
}

// Lambda to match axes names for elementwise additions
// Useful if MXnet has created two identically shaped tensors
// from different paths/names
py::object PyEmitter::match_axes(const py::object& lhs, const py::object& rhs) {
  pyvec lhsaxes;
  pyvec rhsaxes;
  for (auto ax : lhs.attr("axes")) lhsaxes.push_back(ax.cast<py::object>());
  for (auto ax : rhs.attr("axes")) rhsaxes.push_back(ax.cast<py::object>());
  if (lhsaxes.size() != rhsaxes.size()) return rhs;
  for (size_t i = 0; i < lhsaxes.size(); ++i)
    if (lhsaxes[i].attr("length").cast<int>() !=
        rhsaxes[i].attr("length").cast<int>())
      return rhs;

  return ng_.attr("cast_axes")(rhs, lhs.attr("axes"));
}

// unary op genrating function generator
UnaryOps PyEmitter::create_UnaryOps(const py::module& ng) {
  UnaryOps output;
  for (auto op : {"absolute", "exp", "tanh", "sigmoid", "log", "negative",
                  "square", "sign"})
    output[op] = [ng, op](const py::object& py_operand,
                          const std::string& name) {
      return ng.attr(op)(py_operand).attr("named")(name);
    };

  output["_copy"] = [ng](const py::object& py_operand,
                         const std::string& name) { return py_operand; };

  output["relu"] = [ng](const py::object& py_operand, const std::string& name) {
    return ng.attr("maximum")(py_operand, ng.attr("constant")(0.0))
        .attr("named")(name);
  };

  output["softrelu"] = [ng](const py::object& py_operand,
                            const std::string& name) {
    return ng
        .attr("log")(ng.attr("add")(ng.attr("constant")(1.0), ng.attr("exp")))
        .attr("named")(name);
  };

  output["Flatten"] = [ng](const py::object& py_operand,
                           const std::string& name) {
    return ng.attr("flatten_at")(py_operand, num_axes(py_operand) - 1)
        .attr("named")(name);
  };

  return output;
}

// binary op generating function generator
BinaryOps PyEmitter::create_BinaryOps(const py::module& ng) {
  BinaryOps output;
  for (auto op : {"add", "divide", "equal", "greater_equal", "greater",
                  "less_equal", "less", "maximum", "minimum", "multiply",
                  "not_equal", "pow", "mod", "subtract", "dot"})
    output[op] = [ng, op, this](const py::object& lhs, const py::object& rhs,
                                const std::string& name) {
      return ng.attr(op)(lhs, match_axes(lhs, rhs)).attr("named")(name);
    };
  return output;
}

// MXNet high level ops generating function
LayerOps PyEmitter::create_LayerOps(const py::module& ng) {
  LayerOps output;
  output["split"] = [ng, this](const NodePtr& node, py::object data) {

    int axis = 1;
    int num_outputs = 1;
    int index = node->multioutput_index;
    bool squeeze_axis = false;
    for (auto& kv : node->orig_node->attrs.dict) {
      if (kv.first == "num_outputs") num_outputs = std::stoi(kv.second);
      if (kv.first == "axis") axis = std::stoi(kv.second);
      if (kv.first == "squeeze_axis") squeeze_axis = std::stoi(kv.second);
    }

    pyvec axes;
    pyvec slices;
    int i = 0;
    int step = 0;
    for (auto ax : data.attr("axes")) {
      if (i != axis) {
        axes.push_back(ax.cast<py::object>());
        slices.push_back(py::slice{0, ax.attr("length").cast<int>(), 1});
      } else {
        step = ax.attr("length").cast<int>() / num_outputs;
        slices.push_back(py::slice{index * step, (index + 1) * step, 1});
      }
      i+=1;
    }
    py::object op;
    if (squeeze_axis && step == 1) {
      op = ng_.attr("tensor_slice")(data, createPyTuple(slices),
                                    createPyTuple(axes));
    } else {
      op = ng_.attr("tensor_slice")(data, createPyTuple(slices));
    }
    return op;
  };

  output["expand_dims"] = [ng, this](const NodePtr& node, py::object data) {
    int axis = 1;
    for (auto& kv : node->orig_node->attrs.dict)
      if (kv.first == "axis") axis = std::stoi(kv.second);

    auto T = make_axis(1, node->name + "_T");
    return ng.attr("expand_dims")(data, T, axis);
  };

  output["Concat"] = [ng, this](const NodePtr& node, py::object data) {
    int axis = 1;
    for (auto& kv : node->orig_node->attrs.dict)
      if (kv.first == "axis") axis = std::stoi(kv.second);

    auto T = getNthAxis(data, axis);
    py::list inputs;

    auto first = true;
    for (auto n : node->inputs) {
      if (first) {
        inputs.attr("append")(data);
      } else {
        inputs.attr("append")(match_axes(data, op_map[n->name]));
      }
    }
    return ng.attr("concat_along_axis")(inputs, T);
  };
  // Create a fully connected layer op in Ngraph
  output["FullyConnected"] = [ng, this](const NodePtr& node, py::object data) {

    // create a new axis for this layer and store it in a temporary vectcor
    auto newax = make_axis(node->inputs[2]->shape[0], node->name + "_axis");
    auto dataaxis = getNthAxis(data, 1);
    // create weight placeholder
    py::object weight;
    if (op_map.find(node->inputs[1]->name) != op_map.end()) {
      weight = op_map[node->inputs[1]->name];
      weight =
          ng.attr("cast_axes")(weight, createPyTuple(pyvec{newax, dataaxis}));
    } else {
      weight = createPyPlaceholder(node->inputs[1]->name,
                                   createPyTuple(pyvec{newax, dataaxis}));
    }
    // create bias placeholder
    py::object bias;
    if (op_map.find(node->inputs[2]->name) != op_map.end()) {
      bias = op_map[node->inputs[2]->name];
      bias = ng.attr("cast_axes")(bias, createPyTuple(pyvec{newax}));
    } else {
      bias = createPyPlaceholder(node->inputs[2]->name,
                                 createPyTuple(pyvec{newax}));
    }

    auto op = ng.attr("add")(ng.attr("dot")(data, weight), bias)
                  .attr("named")(node->name);
    return op;
  };
  // Bridge Between MXNet's Convolution Op and ngraphs Convolution Op
  output["Convolution"] = [ng, this](const NodePtr& node, py::object data) {
    // Set default ngraph parameters
    py::dict params("str_h"_a = 1, "str_w"_a = 1, "str_d"_a = 1, "pad_h"_a = 0,
                    "pad_w"_a = 0, "pad_d"_a = 0, "dil_h"_a = 1, "dil_w"_a = 1,
                    "dil_d"_a = 1);
    // Parse the mxnet parameters into ngraph's language
    bool no_bias = false;
    int num_filter;
    for (auto& kv : node->orig_node->attrs.dict) {
      if (kv.first == "stride") {
        auto strs = getInts(kv.second);
        params["str_h"] = strs[0];
        params["str_w"] = strs[1];
      } else if (kv.first == "pad") {
        auto pads = getInts(kv.second);
        params["pad_h"] = pads[0];
        params["pad_w"] = pads[1];
      } else if (kv.first == "dilate") {
        auto dils = getInts(kv.second);
        params["dil_h"] = dils[0];
        params["dil_w"] = dils[1];
      } else if (kv.first == "no_bias") {
        if (kv.second == "True" || kv.second == "1") {
          no_bias = true;
        }
      } else if (kv.first == "num_filter") {
        num_filter = std::stoi(kv.second);
      }
    }
    // get data and associated axes
    auto C = getNthAxis(data, 0);
    auto D = getNthAxis(data, 1);
    auto H = getNthAxis(data, 2);
    auto W = getNthAxis(data, 3);
    auto N = getNthAxis(data, 4);
    // get kernel axes
    auto K = make_axis(node->inputs[1]->shape[0], node->name + "_K");
    auto T = make_axis(1, node->name + "_T");
    auto R = make_axis(node->inputs[1]->shape[2], node->name + "_R");
    auto S = make_axis(node->inputs[1]->shape[3], node->name + "_S");
    // get output axes
    auto M = make_axis(1, node->name + "_M");
    auto P = make_axis(node->shape[2], node->name + "_P");
    auto Q = make_axis(node->shape[3], node->name + "_Q");
    // create weight placeholder
    auto weight_in = createPyPlaceholder(node->name + "_weight",
                                         createPyTuple(pyvec{K, C, R, S}));

    // reshape the weight
    auto weight =
        ng.attr("axes_with_order")(ng.attr("expand_dims")(weight_in, T, 2),
                                   createPyTuple(pyvec{C, T, R, S, K}));
    // perform convolution
    auto op = ng.attr("convolution")(params, data, weight,
                                     createPyTuple(pyvec{K, M, P, Q, N}));
    // Add bias placeholder/op if this layer has bias
    if (!no_bias) {
      auto bias =
          createPyPlaceholder(node->name + "_bias", createPyTuple(pyvec{K}));
      op = ng.attr("add")(op, bias);
    }
    return op;

  };
  // Bridge Between MXNet's Pooling Op and ngraphs Pooling Op
  output["Pooling"] = [ng, this](const NodePtr& node, py::object data) {
    // Default pooling parameters
    py::dict params("str_c"_a = 1, "str_h"_a = 1, "str_w"_a = 1, "str_d"_a = 1,
                    "pad_h"_a = 0, "pad_c"_a = 0, "pad_w"_a = 0, "pad_d"_a = 0,
                    "J"_a = 1, "T"_a = 1, "R"_a = 1, "S"_a = 1,
                    "op_a"_a = "avg");
    // Parse MXNet's poling parameters
    // TODO: Assert that the kernel shape = data shape when global_pooling=true
    bool pooling_valid = true;
    bool global_pooling = false;
    auto attrs = node->orig_node->attrs;
    for (auto& kv : attrs.dict) {
      if (kv.first == "pool_type") {
        params["op"] = kv.second;
      } else if (kv.first == "stride") {
        auto strs = getInts(kv.second);
        params["str_h"] = strs[0];
        params["str_w"] = strs[1];
      } else if (kv.first == "pad") {
        auto pads = getInts(kv.second);
        params["pad_h"] = pads[0];
        params["pad_w"] = pads[1];
      } else if (kv.first == "kernel") {
        auto kernels = getInts(kv.second);
        params["R"] = kernels[0];
        params["S"] = kernels[1];
      } else if (kv.first == "global_pool") {
        global_pooling = true;
      } else if (kv.first == "pooling_convention") {
        // TODO: ngraph currently doesn't support "full" pooling.
        // perhaps it can be implemented with padding?
        if (kv.second != "valid") {
          std::cout << ("only valid pooling supported") << std::endl;
          throw;
        }
      }
    }

    auto N = getNthAxis(data, 4);
    auto C = getNthAxis(data, 0);

    auto M = make_axis(1, node->name + "_M");
    auto P = make_axis(node->shape[2], node->name + "_P");
    auto Q = make_axis(node->shape[3], node->name + "_Q");

    // perform pooling
    auto op =
        ng.attr("pooling")(params, data, createPyTuple(pyvec{C, M, P, Q, N}));
    return op;

  };
  // BatchNorm Bridge
  output["BatchNorm"] = [ng, this](const NodePtr& node, py::object data) {
    // Default Batch norm parameters
    float eps = 0.001;
    float momentum = 0.9;
    bool fix_gamma = true;
    float use_global_stats = false;
    int axis = 1;
    // parse mxnet parameters
    auto attrs = node->orig_node->attrs;
    for (auto& kv : attrs.dict) {
      if (kv.first == "eps") {
        eps = std::stof(kv.second);
      } else if (kv.first == "momentum") {
        momentum = std::stof(kv.second);
      } else if (kv.first == "axis") {
        axis = std::stoi(kv.second);
      } else if (kv.first == "fix_gamma") {
        if (kv.second == "False" || kv.second == "0") {
          fix_gamma = false;
        }
      } else if (kv.first == "use_global_stats") {
        if (kv.second == "True" || kv.second == "1") {
          use_global_stats = true;
        }
      }
    }
    // get data, channel axis
    auto C = getNthAxis(data, 0);
    data = ng.attr("flatten_at")(data, 1);
    auto Ctuple = createPyTuple(pyvec{C});
    // create placeholders for batch norm parameters
    auto gamma = createPyPlaceholder(node->name + "_gamma", Ctuple);
    auto beta = createPyPlaceholder(node->name + "_beta", Ctuple);
    // create placeholders for moving averages
    // TODO: Not actually using these anywhere as an auxilary state
    // TODO: Figure out how to pass auxillary states to the right place
    auto moving_mean = createPyPlaceholder(node->name + "_moving_mean", Ctuple);
    auto moving_var = createPyPlaceholder(node->name + "_moving_var", Ctuple);
    // calculate batch mean and variance
    auto mean = ng.attr("mean")(data, "out_axes"_a = Ctuple);
    auto var = ng.attr("variance")(data, "out_axes"_a = Ctuple);
    // Momentum update for moving mean/var. Not currenlty used.
    auto mom_update = [momentum, ng](py::object val, py::object gval) {
      auto first = ng.attr("multiply")(gval, momentum);
      auto second = ng.attr("multiply")(val, 1.0 - momentum);
      return ng.attr("add")(first, second);
    };
    // Utility function for actually computing batch norm
    // separated out for global stats.
    auto batch_norm = [&eps, &data, &gamma, &beta, &ng](py::object mean,
                                                        py::object var) {
      auto denom = ng.attr("reciprocal")(
          ng.attr("sqrt")(ng.attr("add")(var, ng.attr("constant")(eps))));
      auto numer = ng.attr("subtract")(data, mean);
      auto xi = ng.attr("multiply")(numer, denom);
      return ng.attr("add")(ng.attr("multiply")(xi, gamma), beta);
    };
    // TODO: Enable use_global_stats
    auto op = batch_norm(mean, var);
    op = ng.attr("unflatten")(op);

    return op;
  };
  return output;
}
}  // end namespace ngraph
