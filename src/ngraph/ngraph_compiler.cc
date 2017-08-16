#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "reverse_iterate.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>

using namespace pybind11::literals;

namespace ngraph {
// Function to convert TShape object into a list of dimension shapes in
// a python tuple
py::tuple TShapeToTuple(nnvm::TShape shape) {
  py::tuple shape_tuple = py::make_tuple();
  for (auto x : shape) {
    shape_tuple = shape_tuple.attr("__add__")(py::make_tuple(x));
  }
  return shape_tuple;
}

//unary op genrating function generator
UnaryOps PyCompiler::create_UnaryOps(const py::module& ns, const py::module& ng) {
  UnaryOps output;
  for (auto op : {
         "abs", "exp", "tanh", "sigmoid", "relu",
         "log", "negative", "square", "sign",
       } ) {
    output[op] = [ns, op](const py::object & py_operand,
                          const std::string & name
    ) {return ns.attr(op)(py_operand).attr("named")(name);};
  }
  output["flatten"] = [ng](const py::object & py_operand,
                           const std::string & name
  ) {return ng.attr("flatten_at")(py_operand, 1).attr("named")(name);};

  return output;
}
// binary op generating function generator
BinaryOps PyCompiler::create_BinaryOps(const py::module& ns, const py::module& ng) {
  BinaryOps output;
  using namespace pybind11::literals;
  for (auto op : {
         "add", "divide", "equal", "greater_equal", "greater",
         "less_equal", "less", "maximum", "minimum", "multiply",
         "not_equal", "pow", "mod", "subtract"
       } )
    output[op] = [ns, op](const py::object & lhs,
                          const py::object & rhs,
                          const std::string & name
    ) {
    return ns.attr(op)(lhs, rhs).attr("named")(name);
  };
  output["matmul"] = [ns, ng](const py::object & lhs,
                              const py::object & rhs,
                              const std::string & name
  ) {
    return ng.attr("Transpose")(
             ns.attr("matmul")(lhs, rhs, "transpose_b"_a = 1
                              ).attr("named")(name));
  };
  return output;
}
// Compiter initialization
PyCompiler::PyCompiler() {
  // init python
  InitializePython();
  gil_state state;

  // // import python modules
  ng_ = py::module::import("ngraph");
  ns_ = py::module::import(
          "ngraph.frontends.tensorflow.tf_importer.ngraph_shaped");
  ngt_ = py::module::import("ngraph.transformers");
  transformer_ = ngt_.attr("make_transformer")();

  //Create Operation Maps
  NgraphUnaryOps_ = create_UnaryOps(ns_, ng_);
  NgraphBinaryOps_ = create_BinaryOps(ns_, ng_);

  // Find all the valid operation names
  for (auto x : NgraphUnaryOps_) NgraphOps_.emplace_back(x.first);
  for (auto x : NgraphBinaryOps_) NgraphOps_.emplace_back(x.first);
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

// Main compilation function
nnvm::Graph PyCompiler::Compile(
  nnvm::Graph graph,
  std::unordered_map<std::string, nnvm::TShape>& arg_shape_map,
  std::unordered_map<std::string, int>& arg_dtype_map
) {
  gil_state state;
  auto g = ParseNNVMGraph(graph);

  CheckInNGraph(g);

  IdentifySubgraphs(g);

  CollapseSubgraphs(g);

  // Output Graphviz dot files for vizualization
  if (true) {
    g.WriteDot("test.dot");
    for (auto n : g.nodes_) {
      if (n->type == NodeType::kGraph) {
        auto sg = std::dynamic_pointer_cast<Graph>(n);
        std::ostringstream stream;
        stream << "test" << sg->subgraph << ".dot";
        sg->WriteDot(stream.str());
      }
    }
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
          return (n.node->attrs.name == name);};

        for (auto input : node->inputs) {
          auto it = std::find_if(node->inputs.begin(),
                                 node->inputs.end(), matches_name);

          if (it != node->inputs.end()) {
            node->inputs.insert(it, sg_node);
            node->inputs.erase(
              std::remove_if(node->inputs.begin(),
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

// Function to create an nnvm node from a ngraph subgraph
nnvm::NodeEntry PyCompiler::CreateNNVMNode(std::shared_ptr<Graph> graph) {
  // init node, set name
  auto node = nnvm::Node::Create();
  node->attrs.name = graph->name;
  // get the registered operation for the node
  node->attrs.op = get_subgraph_op(graph);
  // setup the ninputs to the node
  for (auto input : graph->inputs)
    node->inputs.emplace_back(
      nnvm::NodeEntry{input->orig_node, 0, 0});
  // create dummy node parameters
  NGraphParam op;
  node->attrs.parsed = std::move(op);

  // init and return NodeEntry
  return nnvm::NodeEntry{node, 0, 0};
}

// compiling a node
void PyCompiler::CompileNode(NodePtr node, std::shared_ptr<Graph> graph) {
  // if the node has been compiled, return
  if (op_map.count(node->name) > 0) {
    return;
    // compile unary operations
  } else if (node->inputs.size() == 1) {
    // if the input hasn't been compiled, compile it
    if (op_map.count(node->inputs[0]->name) == 0) {
      CompileNode(node->inputs[0], graph);
    }
    // get the genrating function for the current operation
    // create the python object for the current operation
    op_map[node->name] = NgraphUnaryOps_[node->operation](
                           op_map[node->inputs[0]->name],
                           node->name);
    // compile binary operations, same idea as unary operations
  } else if (node->inputs.size() == 2) {
    for (int i = 0; i < 2; ++i) {
      if (op_map.count(node->inputs[0]->name) == 0) {
        CompileNode(node->inputs[0], graph);
      }
    }
    op_map[node->name] = NgraphBinaryOps_[node->operation](
                           op_map[node->inputs[0]->name],
                           op_map[node->inputs[1]->name],
                           node->name);
  } else {
    throw ("operation not yet supported");
  }
}

// Compile a Subgraph into ngraph python objects
void PyCompiler::CompileSubgraph(std::shared_ptr<Graph> graph) {
  // initalize a placeholder order vector for this subgraph
  auto subgraph_name = graph->name;
  std::vector<std::string> tmpvec;
  placeholder_order[subgraph_name] = tmpvec;

  for (auto node : graph->nodes_) {
    for (auto input : node->inputs) {
      // find inputs for this node
      auto found_input = std::find(graph->nodes_.begin(),
                                   graph->nodes_.end(),
                                   input);
      // if it's found, create a placeholder python object
      if (found_input == graph->nodes_.end())
        createPyPlaceholder(input, subgraph_name);
    }
  }
  // compile the operations
  for (auto node : graph->nodes_) CompileNode(node, graph);
  // create a python tuple of the variable placeholds to compile the computation
  py::tuple py_placeholders = py::make_tuple();
  for (size_t i = 0; i < placeholder_order[subgraph_name].size(); ++i) {
    py_placeholders = py_placeholders.attr("__add__")(
                        py::make_tuple(
                          op_map[placeholder_order[subgraph_name][int(i)]]));
  }
  // compile the python computation
  graph->py_computation.reset(new py::object(
                                transformer_.attr("computation")(
                                  op_map[graph->nodes_.back()->name],
                                  *py_placeholders)));


  // backward computation
  py::tuple py_deriv_ops = py::make_tuple();
  py::tuple py_back_placeholders = py::make_tuple();
  py::tuple py_shape = TShapeToTuple(graph->shape);

  auto back_grad = ns_.attr("placeholder")("shape"_a = py_shape).attr(
                     "named")(graph->name + "_out_grad");

  for (size_t i = 0; i < placeholder_order[subgraph_name].size(); ++i) {
    py_back_placeholders = py_back_placeholders.attr("__add__")(
                             py::make_tuple(
                               op_map[placeholder_order[subgraph_name][int(i)]]
                             ));
    py_deriv_ops = py_deriv_ops.attr("__add__")(
                     py::make_tuple(
                       ng_.attr("deriv")(
                         op_map[graph->nodes_.back()->name],
                         op_map[placeholder_order[subgraph_name][int(i)]],
                         back_grad)
                     ));
  }

  py_back_placeholders = py_back_placeholders.attr("__add__")(
                           py::make_tuple(back_grad));

  // compile the backward computation
  graph->py_backward.reset(new py::object(
                                transformer_.attr("computation")(
                                  py_deriv_ops,
                                  *py_back_placeholders)));

}

// Function to collapse the intermediary graph into a graph
// with subgraphs for nodes
void PyCompiler::CollapseSubgraphs(Graph& graph) {
  // loop variable for undefined number of subgraphs
  int i = 1;
  while (true) {
    auto tmpGraph = std::make_shared<Graph>();
    // loop over all nodes and add nodes in the current subgraph to
    for (auto node : graph.nodes_)
      if (node->subgraph == i)
        tmpGraph->AddNode(node);

    if (tmpGraph->nodes_.size() == 0) {
      // if we don't find any nodes, assume we've run out of subgraphs
      break;
    } else {
      // if we found nodes, setup subgraph
      tmpGraph->in_ngraph = true;
      tmpGraph->subgraph = i;
      // set node name and shape based on last node in the subgraph
      auto name = tmpGraph->nodes_.back()->name;
      auto shape = tmpGraph->nodes_.back()->shape;
      tmpGraph->name = "subgraph_" + name;
      tmpGraph->shape = shape;
      tmpGraph->dtype = tmpGraph->nodes_.back()->dtype;
      // setup inputs to this subgraph (as a node)
      for (auto node : tmpGraph->nodes_) {
        for (auto input : node->inputs) {
          if (input->subgraph != i)
            tmpGraph->inputs.emplace_back(input);
        }
      }
      // find the position we're replacing in the graph
      auto it = std::find_if(graph.nodes_.begin(), graph.nodes_.end(),
                             [name](NodePtr n) -> bool {return (n->name == name);});
      // insert the subgraph as a node
      graph.nodes_.insert(it, tmpGraph);
      // delete all the ndoes we're replacing with the subgraph
      graph.nodes_.erase(
        std::remove_if(graph.nodes_.begin(),
                       graph.nodes_.end(),
      [i](NodePtr n) -> bool {
        return ((n->subgraph == i) &&
        (n->type == NodeType::kOp));}  ),
      graph.nodes_.end());

      // set subgraph as input to all of the nodes downline.
      for (auto n : graph.nodes_)
        for (size_t i = 0; i < n->inputs.size(); ++i)
          if (n->inputs[i]->name == name)
            n->inputs[i] = tmpGraph;

    }
    i += 1;
  }
}

// function to identify and label connected ngraph ops as subgraphs
void PyCompiler::IdentifySubgraphs(Graph& graph) {
  int sg = 1;
  // loop over the nodes from the back
  for (auto i : reverse_iterate(graph.nodes_)) {
    if (i->subgraph == 0) {
      // select nodes in the a subgraph starting here and going up the graph
      auto subgraph_nodes = graph.DFSselect(i,
      [](NodePtr s) {return s->in_ngraph;});
      // if we found a significantly large subgraph, label it
      if (subgraph_nodes.size() > 2) {
        for (auto node : subgraph_nodes)
          if (node->type == NodeType::kOp)
            node->subgraph = sg;
        sg += 1;
      }
    }
  }
}

// Function for emitting a variable placeholder in ngraph
void PyCompiler::createPyPlaceholder(NodePtr node, std::string subgraph_name) {
  // check if node has already been created in python ngraph
  if (op_map.count(node->name) == 0) {
    // get shape
    py::tuple py_shape = TShapeToTuple(node->shape);
    // create placeholder in python, store it in class dictionary
    op_map[node->name] = ns_.attr("placeholder")("shape"_a = py_shape
                                                ).attr("named")(node->name);
    // store placeholder order in vector for correct execution later
    placeholder_order[subgraph_name].emplace_back(node->name);
  }
}

} //end namespace ngraph
