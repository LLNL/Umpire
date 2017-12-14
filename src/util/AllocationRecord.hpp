#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>

#include <memory>

namespace umpire {

namespace strategy {
  class AllocationStrategy;
}

namespace util {

struct AllocationRecord
{
  void* m_ptr;
  size_t m_size;
  std::shared_ptr<strategy::AllocationStrategy> m_strategy;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
