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

  uintptr_t record = m_records.remove(reinterpret_cast<uintptr_t>(ptr));
  return reinterpret_cast<AllocationRecord*>(record);
  //return nullptr;
}

AllocationRecord*
AllocationMap::find(void* ptr)
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
  AddressPair record = 
    m_records.atOrBefore(reinterpret_cast<uintptr_t>(ptr));

  void* parent_ptr = reinterpret_cast<void*>(record.key);
  AllocationRecord* alloc_record = reinterpret_cast<AllocationRecord*>(record.value);

  if (alloc_record && ((static_cast<char*>(parent_ptr) + alloc_record->m_size) >= static_cast<char*>(ptr))) {
    UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr << " with size " << alloc_record->m_size);
    return alloc_record;
  } else {
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

} // end of namespace util
} // end of namespace umpire
