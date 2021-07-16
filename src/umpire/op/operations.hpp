#pragma once

namespace umpire {
namespace op {

struct operation {
  static constexpr int arity = -1;
};

template<typename Src, typename Dst>
struct copy : public operation {
  static constexpr int arity = 2;
};

template<typename Src>
struct memset : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct reallocate : public operation {
  static constexpr int arity = 1;
};

}
}