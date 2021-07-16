//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

// allocation_map is a multimap of addresses to addresses. It uses Judy
// for the map, with an array-like object to hold multiple values with
// the same key.

#include "umpire/allocation_record.hpp"
#include "umpire/detail/memory_map.hpp"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <functional>

namespace umpire {
namespace detail {

class allocation_map
{
  class record_list
  {
  public:
    template <typename T>
    struct block
    {
      T rec;
      block* prev;
    };

    using record_block = block<allocation_record>;

    // Iterator for record_list
    class const_iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = allocation_record;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type*;
      using reference = value_type&;

      const_iterator();
      const_iterator(const record_list* list, iterator_begin);
      const_iterator(const record_list* list, iterator_end);
      const_iterator(const const_iterator&) = default;

      const allocation_record& operator*();
      const allocation_record* operator->();
      const_iterator& operator++();
      const_iterator operator++(int);

      bool operator==(const const_iterator& other) const;
      bool operator!=(const const_iterator& other) const;

    private:
      const record_list *m_list;
      record_block* m_curr;
    };

    record_list(allocation_map& map, allocation_record record);
    ~record_list();

    void push_back(const allocation_record& rec);
    allocation_record pop_back();

    const_iterator begin() const;
    const_iterator end() const;

    std::size_t size() const;
    bool empty() const;
    allocation_record* back();
    const allocation_record* back() const;

  private:
    allocation_map& m_map;
    record_block* m_tail;
    std::size_t m_length;
  };

public:
  using map = memory_map<record_list>;

  // Iterator that flattens MemoryMap and record_list iterators
  class const_iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = allocation_record;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    // Iterator(allocation_map* map, const OuterIterType& outer_iter, const InnerIterType& inner_iter);
    const_iterator(const allocation_map* map, iterator_begin);
    const_iterator(const allocation_map* map, iterator_end);
    const_iterator(const const_iterator&) = default;

    const allocation_record& operator*();
    const allocation_record* operator->();
    const_iterator& operator++();
    const_iterator operator++(int);

    bool operator==(const const_iterator& other) const;
    bool operator!=(const const_iterator& other) const;

  private:
    using outer_iter = map::ConstIterator;
    using inner_iter = record_list::const_iterator;

    outer_iter m_outer_iter;
    inner_iter m_inner_iter;
    inner_iter m_inner_end;
    outer_iter m_outer_end;
  };

  allocation_map();

  // Would require a deep copy of the Judy data
  allocation_map(const allocation_map&) = delete;

  // Insert a new record -- copies record
  void insert(void* ptr, allocation_record record);

  // Find a record -- throws an exception if the record is not found.
  // allocation_record addresses will not change once registered, so
  // the resulting address of a find(ptr) call can be stored
  // externally until deregistered. Note also that this class
  // deallocates the allocation_record when removed(), so the pointer
  // will become invalid at that point.
  const allocation_record* find(void* ptr) const;
  allocation_record* find(void* ptr);

  // This version of find never throws an exception
  const allocation_record* findRecord(void* ptr) const noexcept;
  allocation_record* findRecord(void* ptr) noexcept;

  // Only allows erasing the last inserted entry for key = ptr
  allocation_record remove(void* ptr);

  // Check if a pointer has been added to the map.
  bool contains(void* ptr) const;

  // Clear all records from the map
  void clear();

  // Returns number of entries
  std::size_t size() const;

  // Print methods -- either matching a predicate or all records
  void print(const std::function<bool (const allocation_record&)>&& predicate,
             std::ostream& os = std::cout) const;

  void printAll(std::ostream& os = std::cout) const;

  const_iterator begin() const;
  const_iterator end() const;

private:
  // Content of findRecord(void*) without the lock
  const allocation_record* doFindRecord(void* ptr) const noexcept;

  // This block pool is used inside record_list, but is needed here so its
  // destruction is linked to that of allocation_map
  fixed_malloc_pool m_block_pool;

  map m_map;
  std::size_t m_size;
  mutable std::mutex m_mutex;
};

} // end of namespace util
} // end of namespace umpire
