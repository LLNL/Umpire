#include "umpire/alloc/Pool.hpp"

namespace umpire {
namespace alloc {

template <typename T>
Pool<T>::Pool(space::MemorySpace* space) :
  m_space(space)
{
  for (int i = 0; i < 32; ++i) {
    m_pointers[i] = nullptr;
    m_lengths[i] = 0;
  }
}

template <typename T>
void*
Pool<T>::allocate(size_t bytes)
{
  T* ptr = nullptr;
  for (int i = 0; i < 32; ++i) {
     if (m_lengths[i] == bytes) {
        m_lengths[i] = -m_lengths[i] ;
        ptr = m_pointers[i] ;
        break ;
     } else if (m_lengths[i] == 0) {
        m_lengths[i] = -static_cast<int>(bytes) ;
        m_pointers[i] = malloc(sizeof(T) * bytes) ;
        ptr = m_pointers[i] ;
        break ;
     }
  }

  //m_space->registerAllocation(ptr, this);
  return ptr;
}

template <typename T>
void
Pool<T>::free(void* ptr)
{
  for (int i = 0; i < 32; ++i) {
    if (m_pointers[i] == ptr) {
      m_lengths[i] = -m_lengths[i];
      ptr = nullptr;
      break;
    }
  }
}

} // end of namespace alloc
} // end of namespace pool
