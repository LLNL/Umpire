#ifndef UMPIRE_AllocatorTraits_HPP
#define UMPIRE_AllocatorTraits_HPP

namespace umpire {
namespace strategy {

struct AllocatorTraits {
  size_t m_initial_size;
  size_t m_maximum_size;
  size_t m_number_allocations;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocatorTraits_HPP
