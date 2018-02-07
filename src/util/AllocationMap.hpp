#ifndef UMPIRE_AllocationMap_HPP
#define UMPIRE_AllocationMap_HPP

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/tpl/judy/judyLArray.h"

namespace umpire {
namespace util {

class AllocationMap
{
  public:
    using AddressPair = judyLArray<uintptr_t, uintptr_t>::pair;

  AllocationMap();
  ~AllocationMap();

  void
  insert(void* ptr, AllocationRecord* record);

  void
  remove(void* ptr);

  AllocationRecord*
  find(void* ptr);

  void
    reset();

  private:
    judyLArray<uintptr_t, uintptr_t> m_records;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
