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
  UMPIRE_LOG(Debug, "Inserting " << ptr);

  m_records.insert(
      reinterpret_cast<uintptr_t>(ptr),
      reinterpret_cast<uintptr_t>(alloc_record));
}

void
AllocationMap::remove(void* ptr)
{
  UMPIRE_LOG(Debug, "Removing " << ptr);

  m_records.remove(reinterpret_cast<uintptr_t>(ptr));
}

AllocationRecord*
AllocationMap::findRecord(void* ptr)
{
  AddressPair record = 
    m_records.atOrBefore(reinterpret_cast<uintptr_t>(ptr));

  void* parent_ptr = reinterpret_cast<void*>(record.key);
  AllocationRecord* alloc_record = reinterpret_cast<AllocationRecord*>(record.value);

  if (alloc_record && ((static_cast<char*>(parent_ptr) + alloc_record->m_size) >= static_cast<char*>(ptr))) {
    UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr << " with size " << alloc_record->m_size);
    return alloc_record;
  } else {
    return nullptr;
  }
}

AllocationRecord*
AllocationMap::find(void* ptr)
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);

  AllocationRecord* alloc_record = findRecord(ptr);

  if (alloc_record) {
    return alloc_record;
  } else {
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

bool
AllocationMap::contains(void* ptr)
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);

  AllocationRecord* alloc_record = findRecord(ptr);

  return (alloc_record != nullptr);
}

} // end of namespace util
} // end of namespace umpire
