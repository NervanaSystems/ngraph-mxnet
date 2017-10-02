#ifndef NGRAPH_PYCOMPILER_UTILS_H_
#define NGRAPH_PYCOMPILER_UTILS_H_

#include <sstream>
#include <iostream>
#include <vector>

namespace ngraph_bridge {

// parse a list like (1, 2, 3) into a vector of ints [1,2,3]
inline std::vector<int> getInts(std::string input) {
  input = input.substr(1, input.size() - 2);
  std::stringstream ss(input);
  std::vector<int> vect;
  int i;
  while (ss >> i) {
    vect.push_back(i);

    if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
  }
  return vect;
}

}  // namespace ngraph
#endif  // UTILS_H_