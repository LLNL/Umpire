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

// AllocationMap is a multimap of addresses to addresses. It uses Judy
// for the map, with an array-like object to hold multiple values with
// the same key.

#include <cstddef>
#include <mutex>
#include <iostream>
#include <iterator>

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/tpl/judy/judy.h"

namespace umpire {
namespace util {

class AllocationMap;
class RecordList;
class RecordListConstIterator;

class AllocationMapConstIterator : public std::iterator<std::forward_iterator_tag, AllocationRecord>
{
  public:
    AllocationMapConstIterator(const AllocationMap* map, bool end);
    AllocationMapConstIterator(const AllocationMap* map, uintptr_t ptr);
    AllocationMapConstIterator(const AllocationMapConstIterator& other) = default;
    ~AllocationMapConstIterator();

    const AllocationRecord& operator*() const;
    const AllocationRecord* operator->() const;
    AllocationMapConstIterator& operator++();
    AllocationMapConstIterator operator++(int);

    bool operator==(const AllocationMapConstIterator& other);
    bool operator!=(const AllocationMapConstIterator& other);
  private:
    Judy* m_array;
    JudySlot* m_last;
    uintptr_t m_ptr;
    RecordListConstIterator* m_iter;
};

class AllocationMap
{
  public:
    // Friend the iterator class
    friend class AllocationMapConstIterator;

    AllocationMap();
    ~AllocationMap();

    // Would require a deep copy of the Judy data
    AllocationMap(const AllocationMap&) = delete;

    // Insert a new record -- copies record
    void insert(void* ptr, AllocationRecord record);

    // Find a record -- throws an exception of the record is not found.
    // AllocationRecord addresses will not change once registered, so
    // the resulting address of a find(ptr) call can be stored
    // externally until deregistered. Note also that this class
    // deallocates the AllocationRecord when removed(), so the pointer
    // will become invalid at that point.
    const AllocationRecord* find(void* ptr) const;
    AllocationRecord* find(void* ptr);

    // This version of find never throws an exception
    const AllocationRecord* findRecord(void* ptr) const;
    AllocationRecord* findRecord(void* ptr);

    // Only allows erasing the last inserted entry for key = ptr
    AllocationRecord remove(void* ptr);

    // Check if a pointer has been added to the map.
    bool contains(void* ptr) const;

    // Clear all records from the map
    void clear();

    // Returns number of entries
    size_t size() const;

    // Print methods -- either matching a predicate or all records
    void print(const std::function<bool (const AllocationRecord&)>&& predicate,
               std::ostream& os = std::cout) const;

    void printAll(std::ostream& os = std::cout) const;

    // Const iterator
    AllocationMapConstIterator begin() const;
    AllocationMapConstIterator end() const;

  private:
    Judy* m_array;
    mutable JudySlot* m_last; // last found value in Judy
    unsigned int m_max_levels, m_depth;
    size_t m_size;
    std::mutex* m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
