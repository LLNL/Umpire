#include "umpire/memory.hpp"

namespace umpire {

std::size_t
memory::get_current_size() const noexcept
{
  return current_size_;
}

std::size_t
memory::get_actual_size() const noexcept
{
  return actual_size_;
}

std::size_t
memory::get_highwatermark() const noexcept
{
  return highwatermark_;
}

}