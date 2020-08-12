//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationMap_HPP
#define UMPIRE_AllocationMap_HPP

// AllocationMap is a multimap of addresses to addresses. It uses Judy
// for the map, with an array-like object to hold multiple values with
// the same key.

#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/MemoryMap.hpp"

namespace umpire {
namespace util {

class AllocationMap {
  class RecordList {
   public:
    template <typename T>
    struct Block {
      T rec;
      Block* prev;
    };

    using RecordBlock = Block<AllocationRecord>;

    // Iterator for RecordList
    class ConstIterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = AllocationRecord;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type*;
      using reference = value_type&;

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
      const RecordList* m_list;
      RecordBlock* m_curr;
    };

    RecordList(AllocationMap& map, AllocationRecord record);
    ~RecordList();

    void push_back(const AllocationRecord& rec);
    AllocationRecord pop_back();

    ConstIterator begin() const;
    ConstIterator end() const;

    std::size_t size() const;
    bool empty() const;
    AllocationRecord* back();
    const AllocationRecord* back() const;

   private:
    AllocationMap& m_map;
    RecordBlock* m_tail;
    std::size_t m_length;
  };

 public:
  using Map = MemoryMap<RecordList>;

  // Iterator that flattens MemoryMap and RecordList iterators
  class ConstIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = AllocationRecord;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    // Iterator(AllocationMap* map, const OuterIterType& outer_iter, const
    // InnerIterType& inner_iter);
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
  std::size_t size() const;

  // Print methods -- either matching a predicate or all records
  void print(const std::function<bool(const AllocationRecord&)>&& predicate,
             std::ostream& os = std::cout) const;

  void printAll(std::ostream& os = std::cout) const;

  ConstIterator begin() const;
  ConstIterator end() const;

 private:
  // Content of findRecord(void*) without the lock
  const AllocationRecord* doFindRecord(void* ptr) const noexcept;

  // This block pool is used inside RecordList, but is needed here so its
  // destruction is linked to that of AllocationMap
  FixedMallocPool m_block_pool;

  Map m_map;
  std::size_t m_size;
  mutable std::mutex m_mutex;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationMap_HPP
