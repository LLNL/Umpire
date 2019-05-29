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

// AllocationMap is a multimap of addresses to addresses. It uses Judy
// for the map, with an array-like object to hold multiple values with
// the same key.

#include <cstddef>
#include <mutex>
#include <iostream>

#include "umpire/util/AllocationRecord.hpp"

// TODO Forward declare Judy and JudySlot
#include "umpire/tpl/judy/judy.h"

namespace umpire {
namespace util {

struct AllocationMap
{
public:
  AllocationMap();
  ~AllocationMap();

  // Would require a deep copy of the Judy data
  AllocationMap(const AllocationMap&) = delete;

  void insert(void* ptr, AllocationRecord record);

  // Find a record -- throws an exception of the record is not found
  AllocationRecord* find(void* ptr) const;

  // This version of find never throws an exception
  AllocationRecord* findRecord(void* ptr) const;

  // Only allows erasing the last inserted entry for key = ptr
  AllocationRecord remove(void* ptr);

  // Check if a pointer has been added to the map
  bool contains(void* ptr) const;

  // Clear all records from the map
  void clear();

  // Print methods -- either matching a predicate or all records
  void print(const std::function<bool (const AllocationRecord&)>&& predicate,
             std::ostream& os = std::cout) const;

  void printAll(std::ostream& os = std::cout) const;

private:
  Judy* m_array;
  mutable JudySlot* m_last; // last found value in Judy
  unsigned int m_max_levels, m_depth;

  // TODO remove pointer
  std::mutex* m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MemoryMap_HPP
