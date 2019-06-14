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

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/util/MemoryMap.hpp"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <functional>

namespace umpire {
namespace util {

class RecordList
{
public:
  template <typename T>
  struct Block
  {
    T rec;
    Block* prev;
  };

  using BlockType = Block<AllocationRecord>;

  // Iterator for RecordList
  class ConstIterator : public std::iterator<std::forward_iterator_tag, AllocationRecord>
  {
  public:
    ConstIterator();
    ConstIterator(const RecordList* list, iterator_begin);
    ConstIterator(const RecordList* list, iterator_end);
    ConstIterator(const ConstIterator&) = default;

    const AllocationRecord& operator*();
    const AllocationRecord* operator->();
    ConstIterator& operator++();
    ConstIterator operator++(int);

    bool operator==(const ConstIterator& other) const;
    bool operator!=(const ConstIterator& other) const;

  private:
    const RecordList *m_list;
    BlockType* m_curr;
  };

  RecordList(AllocationRecord record);
  ~RecordList();

  void push_back(const AllocationRecord& rec);
  AllocationRecord pop_back();

  ConstIterator begin() const;
  ConstIterator end() const;

  size_t size() const;
  bool empty() const;
  AllocationRecord* back();
  const AllocationRecord* back() const;

private:
  BlockType* m_tail;
  size_t m_length;
};

class AllocationMap
{
public:
  using Map = MemoryMap<RecordList>;

  // Iterator that flattens MemoryMap and RecordList iterators
  class ConstIterator : public std::iterator<std::forward_iterator_tag, AllocationRecord>
  {
  public:
    // Iterator(AllocationMap* map, const OuterIterType& outer_iter, const InnerIterType& inner_iter);
    ConstIterator(const AllocationMap* map, iterator_begin);
    ConstIterator(const AllocationMap* map, iterator_end);
    ConstIterator(const ConstIterator&) = default;

    const AllocationRecord& operator*();
    const AllocationRecord* operator->();
    ConstIterator& operator++();
    ConstIterator operator++(int);

    bool operator==(const ConstIterator& other) const;
    bool operator!=(const ConstIterator& other) const;

  private:
    using OuterIter = Map::ConstIterator;
    using InnerIter = RecordList::ConstIterator;

    OuterIter m_outer_iter;
    InnerIter m_inner_iter;
    InnerIter m_inner_end;
    OuterIter m_outer_end;
  };

  AllocationMap();

  // Would require a deep copy of the Judy data
  AllocationMap(const AllocationMap&) = delete;

  // Insert a new record -- copies record
  void insert(void* ptr, AllocationRecord record);

  // Find a record -- throws an exception if the record is not found.
  // AllocationRecord addresses will not change once registered, so
  // the resulting address of a find(ptr) call can be stored
  // externally until deregistered. Note also that this class
  // deallocates the AllocationRecord when removed(), so the pointer
  // will become invalid at that point.
  const AllocationRecord* find(void* ptr) const;
  AllocationRecord* find(void* ptr);

  // This version of find never throws an exception
  const AllocationRecord* findRecord(void* ptr) const noexcept;
  AllocationRecord* findRecord(void* ptr) noexcept;

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

  ConstIterator begin() const;
  ConstIterator end() const;

private:
  // Content of findRecord(void*) without the lock
  const AllocationRecord* doFindRecord(void* ptr) const noexcept;

  Map m_map;
  size_t m_size;
  mutable std::mutex m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
