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
#ifndef UMPIRE_AllocationMap_HPP
#define UMPIRE_AllocationMap_HPP

#include "umpire/util/AllocationRecord.hpp"

#include <cstdint>
#include <mutex>

#include "umpire/tpl/judy/judyL2Array.h"

namespace umpire {
namespace util {

class AllocationMap
{
  public:
    using AddressPair = judyL2Array<uintptr_t, uintptr_t>::cpair;
    using EntryVector = judyL2Array<uintptr_t, uintptr_t>::vector;
    using Entry = AllocationRecord*;

  AllocationMap();
  ~AllocationMap();

  void
  insert(void* ptr, AllocationRecord* record);

  AllocationRecord*
  remove(void* ptr);

  AllocationRecord*
  find(void* ptr);

  bool
  contains(void* ptr);

  void
    reset();


  void
    printAll();

  private:
    AllocationRecord* findRecord(void* ptr);

    judyL2Array<uintptr_t, uintptr_t> m_records;

    std::mutex* m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
