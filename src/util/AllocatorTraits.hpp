#ifndef UMPIRE_AllocatorTraits_HPP
#define UMPIRE_AllocatorTraits_HPP

namespace umpire {

struct AllocatorTraits {
  bool m_pool;
  size_t m_initial_size;
  size_t m_maximum_size;
  size_t m_slots;
};

} // end of namespace umpire

#endif // UMPIRE_AllocatorTraits_HPP
