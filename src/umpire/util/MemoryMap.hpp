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

#include "umpire/tpl/judy/judy.h"

#include "umpire/util/FixedMallocPool.hpp"

#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>

namespace umpire {
namespace util {

struct iterator_begin {};
struct iterator_end {};

// MemoryMap maps addresses to a templated type Value, using a
// FixedMallocPool underneath for speed. It is not threadsafe.

template <typename V>
class MemoryMap
{
public:
  using Key = void*;
  using Value = V;
  using KeyValuePair = std::pair<Key, Value*>;

  template <bool Const = false>
  class Iterator : public std::iterator<std::forward_iterator_tag, Value> {
  public:

    template <bool OtherConst> friend class Iterator;

    using Map = typename std::conditional<Const, const MemoryMap<Value>, MemoryMap<Value>>::type;
    using ValuePtr = typename std::conditional<Const, const Value*, Value*>::type;

    using Content = std::pair<Key, ValuePtr>;
    using Reference = typename std::conditional<Const, const Content&, Content&>::type;
    using Pointer = typename std::conditional<Const, const Content*, Content*>::type;

    Iterator(Map* map);
    Iterator(Map* map, iterator_begin);
    Iterator(Map* map, iterator_end);

    template<bool OtherConst>
    Iterator(const Iterator<OtherConst>& other);

    Reference operator*();
    Pointer operator->();
    Iterator& operator++();
    Iterator operator++(int);

    template <bool OtherConst>
    bool operator==(const Iterator<OtherConst>& other) const;

    template <bool OtherConst>
    bool operator!=(const Iterator<OtherConst>& other) const;

  private:
    Map* m_map;
    Content m_pair;
  };

  template <bool Const> friend class Iterator;

  MemoryMap();
  ~MemoryMap();

  // Would require a deep copy of the Judy data
  MemoryMap(const MemoryMap&) = delete;

  // Return pointer-to or emplaces a new Value with args to the constructor
  template <typename... Args>
  std::pair<Iterator<false>, bool> get(void* ptr, Args&&... args) noexcept;

  // Insert a new Value at ptr
  Iterator<false> insert(void* ptr, const Value& val);

  // Find a value -- returns what would be the entry immediately before ptr
  Iterator<true> findOrBefore(void* ptr) const noexcept;
  Iterator<false> findOrBefore(void* ptr) noexcept;

  // Find a value -- returns end() if not found
  Iterator<true> find(void* ptr) const noexcept;
  Iterator<false> find(void* ptr) noexcept;

  // Iterators
  Iterator<true> begin() const;
  Iterator<false> begin();

  Iterator<true> end() const;
  Iterator<false> end();

  // Remove the entry at ptr
  void remove(void* ptr);

  // Remove/Deallocate the last found entry
  // WARNING: Use this with caution. remove(void*) is safer, but
  // requires an additional lookup and does not return the contents.
  void removeLast();

  // Clear all entries
  void clear();

  // Number of entries
  size_t size() const noexcept;

private:
  // Helper method for public findOrBefore()
  void doFindOrBefore(void* ptr) const noexcept;

  mutable Judy* m_array;    // Judy array
  mutable JudySlot* m_last; // Last found value in judy array
  mutable uintptr_t m_oper;     // pointer to object that last operated on judy array
  FixedMallocPool m_pool;   // value pool
  size_t m_size;            // number of objects stored
};

} // end of namespace util
} // end of namespace umpire

#include "umpire/util/MemoryMap.inl"

#endif // UMPIRE_MemoryMap_HPP
