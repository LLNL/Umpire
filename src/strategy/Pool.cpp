#include "umpire/alloc/Pool.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

Pool::Pool(std::shared_ptr<umpire::Allocator>& allocator) :
  AllocationStrategy(allocator)
{
  UMPIRE_LOG("Creating pool");
  for (int i = 0; i < 32; ++i) {
    m_pointers[i] = nullptr;
    m_lengths[i] = 0;
  }
}

void*
Pool::allocate(size_t bytes)
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
    //m_space->registerAllocation(ptr, this);
  }

  return ptr;
}

void
Pool<alloc>::deallocate(void* ptr)
{
  for (int i = 0; i < 32; ++i) {
    if (m_pointers[i] == ptr) {
      if (m_space) {
        //m_space->un/registerAllocation(ptr);
      }
      m_lengths[i] = -m_lengths[i];
      ptr = nullptr;
      break;
    }
  }
}

} // end of namespace alloc
} // end of namespace pool
