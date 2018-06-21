//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
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

AllocationRecord*
AllocationMap::remove(void* ptr)
{
  UMPIRE_LOG(Debug, "Removing " << ptr);

  auto record_vector = 
    const_cast<std::vector<uintptr_t>*>(m_records.find(reinterpret_cast<uintptr_t>(ptr)));

  AllocationRecord* ret = nullptr;

  if (record_vector) {
    if (record_vector->size() > 0) {
      ret = reinterpret_cast<AllocationRecord*>(record_vector->back());
      record_vector->pop_back();
      if (record_vector->empty()) {
        m_records.removeEntry(reinterpret_cast<uintptr_t>(ptr));
      }
    } else {
      UMPIRE_ERROR("Cannot remove " << ptr );
    }
  } else {
    UMPIRE_ERROR("Cannot remove " << ptr );
  }

  return ret;
}

AllocationRecord*
AllocationMap::findRecord(void* ptr)
{
  AddressPair record = 
    m_records.atOrBefore(reinterpret_cast<uintptr_t>(ptr));

  if (record.value) {
    void* parent_ptr = reinterpret_cast<void*>(record.key);
    AllocationRecord* alloc_record = reinterpret_cast<AllocationRecord*>(record.value->back());

    if (alloc_record && ((static_cast<char*>(parent_ptr) + alloc_record->m_size) >= static_cast<char*>(ptr))) {
      UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr << " with size " << alloc_record->m_size);
      return alloc_record;
    }
  }

  return nullptr;
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
