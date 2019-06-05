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

// MemoryMap maps addresses to a templated type Value, using a
// FixedMallocPool underneath for speed. It is not threadsafe.

template <typename Value>
class MemoryMap
{
public:
  using KeyType = uintptr_t;
  using ValueType = Value;

  template <bool Const = false>
  class Iterator : public std::iterator<std::forward_iterator_tag, Value> {
  public:
    using Reference = typename std::conditional<Const, Value const&, Value&>::type;
    using Pointer = typename std::conditional<Const, Value const*, Value*>::type;

    Iterator(Judy* array, JudySlot* last, KeyType key);
    Iterator(Judy* array, bool end);
    Iterator(const Iterator& other) = default;

    Reference operator*();
    Pointer operator->();
    Iterator& operator++();
    Iterator operator++(int);

    template <bool OtherConst>
    bool operator==(const Iterator<OtherConst>& other);

    template <bool OtherConst>
    bool operator!=(const Iterator<OtherConst>& other);

  private:
    Judy* m_array;
    JudySlot* m_last;
    KeyType m_key;
  };

  // TODO These should be noexcept
  MemoryMap();
  ~MemoryMap();

  // Would require a deep copy of the Judy data
  MemoryMap(const MemoryMap&) = delete;

  // Return pointer-to or emplaces a new Value with args to the constructor
  template <typename... Args>
  std::pair<Iterator<false>, bool> get(void* ptr, Args&... args);

  // Insert a new Value at ptr
  Iterator<false> insert(void* ptr, const Value& val);

  // Find a value -- returns end() if not found
  Iterator<true> find(void* ptr) const;
  Iterator<false> find(void* ptr);

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

private:
  // Helper method for public find()
  KeyType doFind(void* ptr) const;

  mutable Judy* m_array;
  mutable JudySlot* m_last; // last found value in m_array
  size_t m_size;
  FixedMallocPool m_pool;
};

} // end of namespace util
} // end of namespace umpire

#include "umpire/util/MemoryMap.inl"

#endif // UMPIRE_MemoryMap_HPP
