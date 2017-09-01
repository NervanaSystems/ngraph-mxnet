#include "ngraph_compiler.h"
#include "ngraph_compiler_utils.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <algorithm>
#include "ngraph_nnvm_ops.h"

namespace ngraph {

// Compiter initialization
PyCompiler::PyCompiler() {
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

// Main compilation function
nnvm::Graph PyCompiler::Compile(
    nnvm::Graph graph,
    std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
    std::unordered_map<std::string, int>& arg_dtype_map) {
  gil_state state;
  auto g = ParseNNVMGraph(graph);

  CheckInNGraph(g);

  g.IdentifySubgraphs([](NodePtr s) { return s->in_ngraph; });

  g.CollapseSubgraphs();

  // Output Graphviz dot files for vizualization
  if (false) {
    g.WriteSubgraphDots("test");
  }

  for (auto node : g.nodes_) {
    // store the input variable shape for use by nnvm
    // This is happening because my nnvm graph manipulations are
    // breaking the infer shape functionality, so shapes of inputs
    // don't get properly inferred. Works, because we're inferring
    // the shapes before doing all of this, but not ideal
    if (node->type != NodeType::kGraph) {
      arg_shape_map[node->name] = node->shape;
      arg_dtype_map[node->name] = node->dtype;
    }
  }

  // find the subgraphs
  for (auto n : g.nodes_) {
    if (n->type == NodeType::kGraph) {
      // extract and compile subgraph
      auto sg = std::dynamic_pointer_cast<Graph>(n);
      // compile the subgraph into a python computation
      CompileSubgraph(sg);
      // register compiled subgraph with nnvm
      register_subgraph(sg);
      // create nnvm node
      auto sg_node = CreateNNVMNode(sg);
      // setup nnvm node name
      auto name = sg->nodes_.back()->name;

      // use nnvm depth first search to fix node connections in nnvm
      nnvm::DFSVisit(graph.outputs, [sg_node, &name](const nnvm::NodePtr node) {
        auto matches_name = [&name](nnvm::NodeEntry n) -> bool {
          return (n.node->attrs.name == name);
        };

        for (auto input : node->inputs) {
          auto it = std::find_if(node->inputs.begin(), node->inputs.end(),
                                 matches_name);

          if (it != node->inputs.end()) {
            node->inputs.insert(it, sg_node);
            node->inputs.erase(std::remove_if(node->inputs.begin(),
                                              node->inputs.end(), matches_name),
                               node->inputs.end());
          }
        }
      });
    }
  }

  // create a new output graph
  nnvm::Graph out_graph;

  // initialize it with original graph nodes
  out_graph.outputs = graph.outputs;

  return out_graph;
}

// Check nodes in NGraph
void PyCompiler::CheckInNGraph(Graph& graph) {
  // loop over nodes
  for (auto node : graph.nodes_) {
    // if it's an operation, check operation name
    if (node->type == NodeType::kOp) {
      for (auto op : NgraphOps_) {
        if (node->operation == op) {
          node->in_ngraph = true;
          break;
        }
      }
    } else {
      node->in_ngraph = true;
    }
  }
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
  for (size_t i = 0; i < placeholder_order[subgraph_name].size(); ++i) {
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
      ng_.attr("placeholder")(op_map[graph->nodes_.back()->name].attr("axes"))
          .attr("named")(graph->name + "_out_grad");

  py::tuple py_back_placeholders = py::make_tuple(back_grad);
  for (size_t i = 0; i < placeholder_order[subgraph_name].size(); ++i) {
    py_back_placeholders = py_back_placeholders.attr("__add__")(
        py::make_tuple(op_map[placeholder_order[subgraph_name][int(i)]]));
    py_deriv_ops = py_deriv_ops.attr("__add__")(py::make_tuple(ng_.attr(
        "deriv")(op_map[graph->nodes_.back()->name],
                 op_map[placeholder_order[subgraph_name][int(i)]], back_grad)));
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
      if (getNthAxis(data, 0) !=
          getNthAxis(op_map[graph->nodes_[0]->inputs[0]->name], 0)) {
        data = ng_.attr("Transpose")(data);
      }
    }
    op_map[node->name] =
        NgraphLayerOps_[node->operation](node, data, graph->name);
  } else if (node->inputs.size() == 1) {
    // get the genrating function for the current operation
    // create the python object for the current operation
    op_map[node->name] = NgraphUnaryOps_[node->operation](
        op_map[node->inputs[0]->name], node->name);
    // compile binary operations, same idea as unary operations
  } else if (node->inputs.size() == 2) {
    op_map[node->name] = NgraphBinaryOps_[node->operation](
        op_map[node->inputs[0]->name], op_map[node->inputs[1]->name],
        node->name);
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

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodeEntry PyCompiler::CreateNNVMNode(std::shared_ptr<Graph> graph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = graph->name;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(graph);
  // setup the ninputs to the node
  for (auto input : graph->inputs)
    node->inputs.emplace_back(nnvm::NodeEntry{input->orig_node, 0, 0});
  // create dummy node parameters
  NGraphParam op;
  node->attrs.parsed = std::move(op);

  // init and return NodeEntry
  return nnvm::NodeEntry{node, 0, 0};
}

// Function for emitting a variable placeholder in ngraph
py::object PyCompiler::createPyPlaceholder(std::string name, py::tuple axes) {
  // create placeholder in python, store it in class dictionary
  auto op = ng_.attr("placeholder")(axes).attr("named")(name);
  op_map[name] = op;
  return op;
}

// Utility for creating a named ngraph axis
py::object PyCompiler::make_axis(int length, std::string name) {
  return ng_.attr("make_axis")("length"_a = length, "name"_a = name);
}

// unary op genrating function generator
UnaryOps PyCompiler::create_UnaryOps(const py::module& ng) {
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
BinaryOps PyCompiler::create_BinaryOps(const py::module& ng) {
  BinaryOps output;
  // Lambda to match axes names for elementwise additions
  // Useful if MXnet has created two identically shaped tensors
  // from different paths/names
  auto match_axes = [ng](const py::object& lhs, const py::object& rhs) {
    pyvec lhsaxes;
    pyvec rhsaxes;
    for (auto ax : lhs.attr("axes")) lhsaxes.push_back(ax.cast<py::object>());
    for (auto ax : rhs.attr("axes")) rhsaxes.push_back(ax.cast<py::object>());
    if (lhsaxes.size() != rhsaxes.size()) return rhs;
    for (size_t i = 0; i < lhsaxes.size(); ++i) {
      if (lhsaxes[i].attr("length").cast<int>() !=
          rhsaxes[i].attr("length").cast<int>()) {
        return rhs;
      }
    }
    return ng.attr("cast_axes")(rhs, lhs.attr("axes"));
  };

  for (auto op : {"add", "divide", "equal", "greater_equal", "greater",
                  "less_equal", "less", "maximum", "minimum", "multiply",
                  "not_equal", "pow", "mod", "subtract", "dot"})
    output[op] = [ng, op, match_axes](const py::object& lhs,
                                      const py::object& rhs,
                                      const std::string& name) {
      return ng.attr(op)(lhs, match_axes(lhs, rhs)).attr("named")(name);
    };
  return output;
}
// MXNet high level ops generating function
LayerOps PyCompiler::create_LayerOps(const py::module& ng) {
  LayerOps output;
  // Create a fully connected layer op in Ngraph
  output["FullyConnected"] = [ng, this](const NodePtr& node, py::object data,
                                        const std::string& subgraph_name) {

    // create a new axis for this layer and store it in a temporary vectcor
    auto newax = make_axis(node->inputs[2]->shape[0], node->name + "_axis");
    pyvec weight_ax_vec;
    weight_ax_vec.push_back(newax);
    // get the last axis of the datay
    weight_ax_vec.push_back(getNthAxis(data, 1));

    // create weight placeholder
    auto weight = createPyPlaceholder(node->inputs[1]->name,
                                      createPyTuple(weight_ax_vec));

    // create bias placeholder
    auto bias =
        createPyPlaceholder(node->inputs[2]->name, createPyTuple(pyvec{newax}));

    // return the op
    return ng.attr("add")(ng.attr("dot")(data, weight), bias)
        .attr("named")(node->name);
  };
  // Bridge Between MXNet's Convolution Op and ngraphs Convolution Op
  output["Convolution"] = [ng, this](const NodePtr& node, py::object data,
                                     const std::string& subgraph_name) {
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
  output["Pooling"] = [ng, this](const NodePtr& node, py::object data,
                                 const std::string& subgraph_name) {
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
  output["BatchNorm"] = [ng, this](const NodePtr& node, py::object data,
                                   const std::string& subgraph_name) {
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
    auto batch_norm = [eps, data, gamma, beta, ng](py::object mean,
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

// Generator to create functions that convert mxnet layer operations
// into a series of ngraph operations
layerGraphs PyCompiler::create_layerGraphs() {
  layerGraphs layer_funcs;
  layer_funcs[std::string("Activation")] = [](const NodePtr node) {
    Graph tmpGraph;
    auto act_type = node->orig_node->attrs.dict["act_type"];
    tmpGraph.AddNode(std::make_shared<OpNode>(node->orig_node, node->name,
                                              act_type, node->inputs));
    return tmpGraph;
  };
  return layer_funcs;
}

// Function that parses an nnvm Graph into an intermediary graph
Graph PyCompiler::ParseNNVMGraph(nnvm::Graph& graph) {
  // create inermediary graph
  Graph tmpGraph;
  // Create dictionary of layer->ngraph functions
  auto layer_funcs = create_layerGraphs();
  // Use NNVM's depth first search to trace the tree and construct the
  // intermediary graph
  nnvm::DFSVisit(graph.outputs, [&graph, &tmpGraph,
                                 &layer_funcs](const nnvm::NodePtr node) {
    const auto& idx = graph.indexed_graph();

    const auto& mutable_nodes = idx.mutable_input_nodes();
    const uint32_t nid = idx.node_id(node.get());
    if (mutable_nodes.count(nid) != 0) {
      // add an auxillary node to the graph
      tmpGraph.AddNode(std::make_shared<AuxNode>(node, node->attrs.name));
    } else if (node->is_variable()) {
      // add variable to the graph
      tmpGraph.AddNode(std::make_shared<VariableNode>(node, node->attrs.name));
    } else {
      // create operation node
      auto op_name = clean_opname(node->op()->name);
      auto op_node = std::make_shared<OpNode>(node, node->attrs.name, op_name);
      // setup operation inputs
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = node->inputs[i];
        std::shared_ptr<Node> tmpnode;
        try {
          tmpnode = tmpGraph[e.node->attrs.name];
        } catch (std::string& error) {
          tmpnode = std::make_shared<VariableNode>(node, e.node->attrs.name);
          tmpGraph.AddNode(tmpnode);
        }
        op_node->inputs.emplace_back(tmpnode);
      }
      if (layer_funcs.count(op_node->operation) != 0) {
        // perform layer expansions
        auto tmp = layer_funcs[op_node->operation](op_node);
        for (auto n : tmp.nodes_) tmpGraph.AddNode(n);
      } else {
        // add operation
        tmpGraph.AddNode(op_node);
      }
    }
  });

  // get the shape and data types of all of the nodes
  const auto& idx = graph.indexed_graph();
  const auto inferred_shapes =
      graph.GetAttr<std::vector<nnvm::TShape>>("shape");
  const auto inferred_dtypes = graph.GetAttr<std::vector<int>>("dtype");
  for (auto node : tmpGraph.nodes_) {
    const uint32_t nid = idx.node_id(node->orig_node.get());
    const uint32_t eid = idx.entry_id(nid, 0);
    node->shape = inferred_shapes[eid];
    node->dtype = inferred_dtypes[eid];
  }
  // return intermediary graph
  return tmpGraph;
}

}  // end namespace ngraph
