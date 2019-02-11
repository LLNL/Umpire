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
#ifndef UMPIRE_AllocationMap_HPP
#define UMPIRE_AllocationMap_HPP

#include "umpire/util/AllocationRecord.hpp"

#include <cstdint>
#include <mutex>

template< typename JudyKey, typename JudyValue >
class judyL2Array;

namespace umpire {
namespace util {

class AllocationMap
{
  public:

  AllocationMap();
  ~AllocationMap();

  void
  insert(void* ptr, AllocationRecord* record);

  AllocationRecord*
  remove(void* ptr);

  AllocationRecord*
  find(void* ptr) const;

  bool
  contains(void* ptr);

  void
  reset();

  void
  printAll() const;

  AllocationRecord*
  findRecord(void* ptr) const;

  private:
    judyL2Array<uintptr_t, uintptr_t>* m_records;

    std::mutex* m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
