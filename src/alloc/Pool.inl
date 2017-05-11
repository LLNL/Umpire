#ifndef UMPIRE_Pool_INL
#define UMPIRE_Pool_INL

#include "umpire/alloc/Pool.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

template <typename alloc>
Pool<alloc>::Pool() :
  m_allocator(alloc{}),
  m_space(nullptr)
{
}

template <typename alloc>
Pool<alloc>::Pool(shared_ptr<space::MemorySpace> space) :
  m_allocator(alloc{}),
  m_space(space)
{
  UMPIRE_LOG("Creating pool");
  for (int i = 0; i < 32; ++i) {
    m_pointers[i] = nullptr;
    m_lengths[i] = 0;
  }
}

template <typename alloc>
void*
Pool<alloc>::allocate(size_t bytes)
{
  void* ptr = nullptr;
  for (int i = 0; i < 32; ++i) {
     if (m_lengths[i] == bytes) {
        m_lengths[i] = -m_lengths[i] ;
        ptr = m_pointers[i] ;
        break ;
     } else if (m_lengths[i] == 0) {
        m_lengths[i] = -static_cast<int>(bytes) ;
        m_pointers[i] = m_allocator.allocate(bytes);
        ptr = m_pointers[i] ;
        break ;
     }
  }

  if (m_space) {
    m_space->registerAllocation(ptr, this);
  }

  return ptr;
}

template <typename alloc>
void
Pool<alloc>::free(void* ptr)
{
  for (int i = 0; i < 32; ++i) {
    if (m_pointers[i] == ptr) {
      if (m_space) {
        m_space->unregisterAllocation(ptr);
      }
      m_lengths[i] = -m_lengths[i];
      ptr = nullptr;
      break;
    }
  }
}

} // end of namespace alloc
} // end of namespace pool

#endif // UMPIRE_Pool_INL
