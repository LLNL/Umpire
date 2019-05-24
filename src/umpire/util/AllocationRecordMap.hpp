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
#ifndef UMPIRE_MemoryMap_HPP
#define UMPIRE_MemoryMap_HPP

// MemoryMap is a multimap of addresses to addresses. It uses
// Judy for the map, with a vector-like object to hold multiple values
// with the same key.

#include <cstddef>
#include <mutex>
#include <iostream>

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/tpl/judy/judy.h"

namespace umpire {
namespace util {

struct AllocationRecordMap
{
public:
  AllocationRecordMap();
  ~AllocationRecordMap();
  AllocationRecordMap(const AllocationRecordMap&) = delete;

  void insert(void* ptr, AllocationRecord record);

  AllocationRecord* find(void* ptr) const;

  // Only allows erasing the last inserted entry for key = ptr
  AllocationRecord remove(void* ptr);

  bool contains(void* ptr) const;

  void clear();

  void print(const std::function<bool (const AllocationRecord*)>&& predicate,
             std::ostream& os = std::cout) const;

  void printAll(std::ostream& os = std::cout) const;

private:
  Judy* m_array;
  mutable JudySlot* m_last; // last found value in Judy
  unsigned int m_max_levels, m_depth;

  std::mutex* m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MemoryMap_HPP
