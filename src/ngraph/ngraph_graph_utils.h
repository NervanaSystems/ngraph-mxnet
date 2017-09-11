#ifndef NGRAPH_GRAPH_UTILS_H_
#define NGRAPH_GRAPH_UTILS_H_

#include <random>

namespace ngraph {

//create a random string to avoid subgraph name collisions
inline std::string randomString(const int length = 12) {
  static const char alphabet[] =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789";
  // set up random number generation
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<> dist(
      0, sizeof(alphabet) / sizeof(*alphabet) - 2);
  // create and return string
  std::string str;
  str.reserve(length);
  std::generate_n(std::back_inserter(str), length,
                  [&]() { return alphabet[dist(rng)]; });
  return str;
}

}  // namespace ngraph
#endif  // UTILS_H_