#include "umpire/util/AllocationMap.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

AllocationMap::AllocationMap()
{
}

AllocationMap::~AllocationMap()
{
}

void
AllocationMap::insert(void* ptr, AllocationRecord* alloc_record)
{
  m_records.insert(
      reinterpret_cast<uintptr_t>(ptr),
      reinterpret_cast<uintptr_t>(alloc_record));
}

void
AllocationMap::remove(void* ptr)
{
  m_records.remove(reinterpret_cast<uintptr_t>(ptr));
}

AllocationRecord*
AllocationMap::find(void* ptr)
{
  AddressPair record = 
    m_records.atOrBefore(reinterpret_cast<uintptr_t>(ptr));

  void* parent_ptr = reinterpret_cast<void*>(record.key);
  AllocationRecord* alloc_record = reinterpret_cast<AllocationRecord*>(record.value);

  if (alloc_record && ((parent_ptr + alloc_record->m_size) >= ptr)) {
    return alloc_record;
  } else {
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

} // end of namespace util
} // end of namespace umpire
