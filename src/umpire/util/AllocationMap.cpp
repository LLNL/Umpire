//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#include "umpire/tpl/judy/judyL2Array.h"

#include <sstream>

namespace {
  using AddressPair = judyL2Array<uintptr_t, uintptr_t>::cpair;
  using EntryVector = judyL2Array<uintptr_t, uintptr_t>::vector;
  using Entry = umpire::util::AllocationRecord*;
}

namespace umpire {
namespace util {


AllocationMap::AllocationMap() :
  m_records(new judyL2Array<uintptr_t, uintptr_t>()),
  m_mutex(new std::mutex())
{
}

AllocationMap::~AllocationMap()
{
  delete m_records;
  delete m_mutex;
}

void
AllocationMap::insert(void* ptr, AllocationRecord* alloc_record)
{
  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "Inserting " << ptr);

    m_records->insert(
        reinterpret_cast<uintptr_t>(ptr),
        reinterpret_cast<uintptr_t>(alloc_record));

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
}

AllocationRecord*
AllocationMap::remove(void* ptr)
{
  Entry ret = nullptr;

  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "Removing " << ptr);

    EntryVector* record_vector =
      const_cast<EntryVector*>(
          m_records->find(reinterpret_cast<uintptr_t>(ptr)));

    if (record_vector) {
      if (record_vector->size() > 0) {

        ret = reinterpret_cast<Entry>(record_vector->back());
        record_vector->pop_back();

        if (record_vector->empty()) {
          m_records->removeEntry(reinterpret_cast<uintptr_t>(ptr));
        }
      }
    } else {
      UMPIRE_ERROR("Cannot remove " << ptr );
    }

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return ret;
}

AllocationRecord*
AllocationMap::findRecord(void* ptr) const
{

  Entry alloc_record = nullptr;

  try {
    UMPIRE_LOCK;
    auto record = m_records->atOrBefore(reinterpret_cast<uintptr_t>(ptr));
    if (record.value) {
      void* parent_ptr = reinterpret_cast<void*>(record.key);
      alloc_record = reinterpret_cast<Entry>(record.value->back());

      if (alloc_record &&
          (
           (static_cast<char*>(parent_ptr) + alloc_record->m_size)
               > static_cast<char*>(ptr)
             || 
           static_cast<char*>(parent_ptr) == static_cast<char*>(ptr)
          )
      ) {
         UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr
            << " with size " << alloc_record->m_size);
      }
      else {
         alloc_record = nullptr;
      }
    }
    UMPIRE_UNLOCK;
  }
  catch (...){
    UMPIRE_UNLOCK;
    throw;
  }

  return alloc_record;
}

AllocationRecord*
AllocationMap::find(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);

  auto alloc_record = findRecord(ptr);

  if (alloc_record) {
    return alloc_record;
  } else {
#if !defined(NDEBUG)
    // use this from a debugger to dump the contents of the AllocationMap
    printAll();
#endif
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

bool
AllocationMap::contains(void* ptr)
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);

  return (findRecord(ptr) != nullptr);
}

void
AllocationMap::print(const std::function<bool (const AllocationRecord*)>&& pred,
                     std::ostream& os) const
{
  for (auto record = m_records->begin(); m_records->success(); record=m_records->next()){
    auto addr = record.key;
    auto vec = *record.value;

    std::stringstream ss;
    ss << reinterpret_cast<void*>(addr) << " : {" << std::endl;
    bool any_match = false;
    for (auto const& records : vec) {
      AllocationRecord* tmp = reinterpret_cast<AllocationRecord*>(records);
      if (pred(tmp)) {
        any_match = true;
        ss << "  " << tmp->m_size <<
          " [ " << reinterpret_cast<void*>(addr) <<
          " -- " << reinterpret_cast<void*>(addr+tmp->m_size) <<
          " ] " << std::endl;
      }
    }
    ss << "}" << std::endl;
    if (any_match) { os << ss.str(); }
  }
}

void
AllocationMap::printAll(std::ostream& os) const
{
  os << "🔍 Printing allocation map contents..." << std::endl;

  print([] (const AllocationRecord*) { return true; }, os);

  os << "done." << std::endl;
}

} // end of namespace util
} // end of namespace umpire
