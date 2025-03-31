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

template<typename Src>
struct advise : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct accessed_by : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct preferred_location : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct read_mostly : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct unset_accessed_by : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct unset_preferred_location : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct unset_read_mostly : public operation {
  static constexpr int arity = 1;
};

#if (defined(UMPIRE_ENABLE_HIP) && HIP_VERSION_MAJOR >= 5) || defined(UMPIRE_ENABLE_CUDA)
template<typename Src>
struct coarse_grain : public operation {
  static constexpr int arity = 1;
};

template<typename Src>
struct unset_coarse_grain : public operation {
  static constexpr int arity = 1;
};
#endif

template<typename Src>
struct prefetch : public operation {
  static constexpr int arity = 1;
};

}
}