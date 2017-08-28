#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

namespace umpire {
namespace util {

struct AllocationRecord
{
  void* m_ptr;
  size_t m_size;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
