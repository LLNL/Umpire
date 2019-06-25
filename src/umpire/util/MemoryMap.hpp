//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryMap_HPP
#define UMPIRE_MemoryMap_HPP

#include "umpire/tpl/judy/judy.h"

#include "umpire/util/FixedMallocPool.hpp"

#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>
#include <mutex>

namespace umpire {
namespace util {

// Tags for iterator constructors
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
  class Iterator_ : public std::iterator<std::forward_iterator_tag, Value> {
  public:

    using Map = typename std::conditional<Const, const MemoryMap<Value>, MemoryMap<Value>>::type;
    using ValuePtr = typename std::conditional<Const, const Value*, Value*>::type;

    using Content = std::pair<Key, ValuePtr>;
    using Reference = typename std::conditional<Const, const Content&, Content&>::type;
    using Pointer = typename std::conditional<Const, const Content*, Content*>::type;

    Iterator_(Map* map, Key ptr);
    Iterator_(Map* map, iterator_begin);
    Iterator_(Map* map, iterator_end);

    template<bool OtherConst>
    Iterator_(const Iterator_<OtherConst>& other);

    Reference operator*();
    Pointer operator->();
    Iterator_& operator++();
    Iterator_ operator++(int);

    template <bool OtherConst>
    bool operator==(const Iterator_<OtherConst>& other) const;

    template <bool OtherConst>
    bool operator!=(const Iterator_<OtherConst>& other) const;

  private:
    Map* m_map;
    Content m_pair;
  };

  template <bool Const> friend class Iterator_;

  using Iterator = Iterator_<false>;
  using ConstIterator = Iterator_<true>;

  MemoryMap();
  ~MemoryMap();

  // Would require a deep copy of the Judy data
  MemoryMap(const MemoryMap&) = delete;

  // Return pointer-to or emplaces a new Value with args to the constructor
  template <typename... Args>
  std::pair<Iterator, bool> get(Key ptr, Args&&... args) noexcept;

  // Insert a new Value at ptr
  Iterator insert(Key ptr, const Value& val);

  // Find a value -- returns what would be the entry immediately before ptr
  ConstIterator findOrBefore(Key ptr) const noexcept;
  Iterator findOrBefore(Key ptr) noexcept;

  // Find a value -- returns end() if not found
  ConstIterator find(Key ptr) const noexcept;
  Iterator find(Key ptr) noexcept;

  // Iterators
  ConstIterator begin() const;
  Iterator begin();

  ConstIterator end() const;
  Iterator end();

  // Remove the entry
  void erase(Key ptr);
  void erase(Iterator iter);
  void erase(ConstIterator oter);

  // Remove/Deallocate the internal judy position. WARNING: Use this
  // with caution, only directly after using a method above.
  // erase(Key) is safer, but requires an additional lookup.
  void removeLast();

  // Clear all entries
  void clear() noexcept;

  // Number of entries
  std::size_t size() const noexcept;

private:
  // Helper method for public findOrBefore()
  Key doFindOrBefore(Key ptr) const noexcept;

  mutable Judy* m_array;    // Judy array
  mutable JudySlot* m_last; // last found value in judy array
  mutable uintptr_t m_oper; // address of last object to set internal judy state
  FixedMallocPool m_pool;   // value pool
  std::size_t m_size;            // number of objects stored
};


} // end of namespace util
} // end of namespace umpire

#include "umpire/util/MemoryMap.inl"

#endif // UMPIRE_MemoryMap_HPP
