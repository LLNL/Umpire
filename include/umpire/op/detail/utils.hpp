#pragma once

namespace umpire {
namespace op {
namespace detail {

template <typename T>
inline std::size_t get_size(std::size_t bytes)
{
  if constexpr (std::is_same_v<T, void>)
    return bytes;
  else
    return bytes * sizeof(T);
}

} // namespace detail
} // namespace op
} // namespace umpire
