#ifndef NGRAPH_COMPILER_UTILS_H_
#define NGRAPH_COMPILER_UTILS_H_

#include <string>

namespace ngraph {

// function to remove modifiers frop op names
inline std::string clean_opname(std::string name) {
  for (std::string str : {"elemwise_", "broadcast_"})
    if (name.substr(0, str.size()) == str) name = name.substr(str.size());
  if (name == "_mul") name = "multiply";
  return name;
}

}  // namespace ngraph
#endif  // UTILS_H_