#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "reverse_iterate.h"
#include <nnvm/node.h>
#include <nnvm/pass.h>


namespace ngraph {
UnaryOps PyCompiler::create_UnaryOps(const py::module& ns, const py::module& ng) {
  UnaryOps output;
  for (auto op : {
         "abs", "exp", "tanh", "sigmoid", "relu",
         "log", "negative", "square", "sign",
       } )
    output[op] = [ns, op](const py::object & py_operand,
    const std::string & name) {
    return ns.attr(op)(py_operand).attr("named")(name);
  };
  using namespace pybind11::literals;
  output["flatten"] = [ng](
  const py::object & py_operand, const std::string & name) {
    return ng.attr("flatten_at")(py_operand, 1).attr("named")(name);
  };
  return output;
}

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
    const std::string & name) {
    return ns.attr(op)(lhs, rhs).attr("named")(name);
  };
  output["matmul"] = [ns, ng](const py::object & lhs,
                              const py::object & rhs,
  const std::string & name) {
    return ng.attr("Transpose")(
             ns.attr("matmul")(lhs, rhs, "transpose_b"_a = 1
                              ).attr("named")(name));
  };
  return output;
}

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

void PyCompiler::CheckInNGraph(Graph& graph) {
  for (auto node : graph.nodes_) {
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

nnvm::Graph PyCompiler::Compile(nnvm::Graph graph) {
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

  for (auto n : g.nodes_) {
    if (n->type == NodeType::kGraph) {
      std::cout << n->name << std::endl;
      auto sg = std::dynamic_pointer_cast<Graph>(n);
      CompileSubgraph(sg);
      register_subgraph(sg);
      auto sg_node = CreateNNVMNode(sg);
      auto name = sg->nodes_.back()->name;
      nnvm::DFSVisit(graph.outputs,
      [sg_node, &name](const nnvm::NodePtr node) {
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

  return graph;
}

nnvm::NodeEntry PyCompiler::CreateNNVMNode(std::shared_ptr<Graph> graph) {
  auto node = nnvm::Node::Create();
  node->attrs.name = graph->name;
  node->attrs.op = get_subgraph_op(graph);
  for (auto input : graph->inputs)
    node->inputs.emplace_back(
      nnvm::NodeEntry{input->orig_node, 0, 0});
  return nnvm::NodeEntry{node, 0, 0};
}

void PyCompiler::CompileNode(NodePtr node, std::shared_ptr<Graph> graph) {
  if (op_map.count(node->name) > 0) {
    return;
  } else if (node->inputs.size() == 1) {
    if (op_map.count(node->inputs[0]->name) == 0) {
      CompileNode(node->inputs[0], graph);
    }
    op_map[node->name] = NgraphUnaryOps_[node->operation](
                           op_map[node->inputs[0]->name],
                           node->name);
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

void PyCompiler::CompileSubgraph(std::shared_ptr<Graph> graph) {
  auto subgraph_name = graph->name;
  std::vector<std::string> tmpvec;
  placeholder_order[subgraph_name] = tmpvec;

  for (auto node : graph->nodes_) {
    for (auto input : node->inputs) {
      auto found_input = std::find(graph->nodes_.begin(),
                                   graph->nodes_.end(),
                                   input);
      if (found_input == graph->nodes_.end())
        createPyPlaceholder(input, subgraph_name);
    }
  }
  for (auto node : graph->nodes_) CompileNode(node, graph);

  py::tuple py_placeholders = py::make_tuple();
  for (size_t i = 0; i < placeholder_order[subgraph_name].size(); ++i) {
    py_placeholders = py_placeholders.attr("__add__")(
                        py::make_tuple(op_map[placeholder_order[subgraph_name][int(i)]]));
  }

  graph->py_computation.reset(new py::object(
                                transformer_.attr("computation")(
                                  op_map[graph->nodes_.back()->name], *py_placeholders)));

}

void PyCompiler::CollapseSubgraphs(Graph& graph) {
  int i = 1;
  while (true) {
    auto tmpGraph = std::make_shared<Graph>();
    for (auto node : graph.nodes_)
      if (node->subgraph == i)
        tmpGraph->AddNode(node);

    if (tmpGraph->nodes_.size() > 0) {
      tmpGraph->in_ngraph = true;
      tmpGraph->subgraph = i;
      auto name = tmpGraph->nodes_.back()->name;
      tmpGraph->name = "subgraph_" + name;
      for (auto node : tmpGraph->nodes_) {
        for (auto input : node->inputs) {
          if (input->subgraph != i)
            tmpGraph->inputs.emplace_back(input);
        }
      }

      auto it = std::find_if(graph.nodes_.begin(), graph.nodes_.end(),
                             [name](NodePtr n) -> bool {return (n->name == name);});

      graph.nodes_.insert(it, tmpGraph);
      graph.nodes_.erase(std::remove_if(
                           graph.nodes_.begin(),
                           graph.nodes_.end(),
      [i](NodePtr n) -> bool {
        return ((n->subgraph == i) &&
        (n->type == NodeType::kOp));}),
      graph.nodes_.end());

      for (auto n : graph.nodes_)
        for (size_t i = 0; i < n->inputs.size(); ++i)
          if (n->inputs[i]->name == name)
            n->inputs[i] = tmpGraph;

    } else {
      break;
    }
    i += 1;
  }
}

void PyCompiler::IdentifySubgraphs(Graph& graph) {
  int sg = 1;
  for (auto i : reverse_iterate(graph.nodes_)) {
    if (i->subgraph == 0) {
      auto subgraph_nodes = graph.DFSselect(i,
      [](NodePtr s) {return s->in_ngraph;});
      if (subgraph_nodes.size() > 2) {
        for (auto node : subgraph_nodes)
          if (node->type == NodeType::kOp)
            node->subgraph = sg;
        sg += 1;
      }
    }
  }
}

py::tuple TShapeToTuple(nnvm::TShape shape) {
  py::tuple shape_tuple = py::make_tuple();
  for (auto x : shape) {
    shape_tuple = shape_tuple.attr("__add__")(py::make_tuple(x));
  }
  return shape_tuple;
}

void PyCompiler::createPyPlaceholder(NodePtr node, std::string subgraph_name) {
  if (op_map.count(node->name) == 0) {
    py::tuple py_shape = TShapeToTuple(node->shape);
    using namespace pybind11::literals;
    op_map[node->name] = ns_.attr("placeholder")("shape"_a = py_shape
                                                ).attr("named")(node->name);
    placeholder_order[subgraph_name].emplace_back(node->name);
  }
}

} //end namespace ngraph
