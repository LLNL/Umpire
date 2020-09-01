//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryMap_HPP
#define UMPIRE_MemoryMap_HPP

#include <cstdint>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <utility>

#include "umpire/tpl/judy/judy.h"
#include "umpire/util/FixedMallocPool.hpp"

namespace umpire {
namespace util {

// Tags for iterator constructors
struct iterator_begin {
};
struct iterator_end {
};

/*!
 * \brief A fast replacement for std::map<void*,Value> for a generic Value.
 *
 * This uses FixedMallocPool and Judy arrays and provides forward
 * const and non-const iterators.
 */
template <typename V>
class MemoryMap {
 public:
  using Key = void*;
  using Value = V;
  using KeyValuePair = std::pair<Key, Value*>;

  template <bool Const = false>
  class Iterator_ {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Value;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    using Map = typename std::conditional<Const, const MemoryMap<Value>,
                                          MemoryMap<Value>>::type;
    using ValuePtr =
        typename std::conditional<Const, const Value*, Value*>::type;

    using Content = std::pair<Key, ValuePtr>;
    using Reference =
        typename std::conditional<Const, const Content&, Content&>::type;
    using Pointer =
        typename std::conditional<Const, const Content*, Content*>::type;

    Iterator_(Map* map, Key ptr);
    Iterator_(Map* map, iterator_begin);
    Iterator_(Map* map, iterator_end);

    template <bool OtherConst>
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

  template <bool Const>
  friend class Iterator_;

  using Iterator = Iterator_<false>;
  using ConstIterator = Iterator_<true>;

  MemoryMap();
  ~MemoryMap();

  // Would require a deep copy of the Judy data
  MemoryMap(const MemoryMap&) = delete;

  /*!
   * \brief Insert Value at ptr in the map if ptr does not exist. Uses
   * copy constructor on Value once.
   *
   * \return Pair of iterator position into map and boolean value
   * whether entry was added. The iterator will be set to end() if no
   * insertion was made.
   */
  std::pair<Iterator, bool> insert(Key ptr, const Value& val) noexcept;

  /*!
   * \brief Insert a key-value pair if pair.first does not exist as a
   * key. Must have first and second fields. Calls the first version.
   *
   * \return See alternative version.
   */
  template <typename P>
  std::pair<Iterator, bool> insert(P&& pair) noexcept;

  /*!
   * \brief Emplaces a new value at ptr in the map, forwarding args to
   * the placement new constructor.
   *
   * \return See alternative version.
   */
  template <typename... Args>
  std::pair<Iterator, bool> insert(Key ptr, Args&&... args) noexcept;

  /*!
   * \brief Find a value at ptr.
   *
   * \return iterator into map at ptr or preceeding position.
   */
  Iterator findOrBefore(Key ptr) noexcept;
  ConstIterator findOrBefore(Key ptr) const noexcept;

  /*!
   * \brief Find a value at ptr.
   *
   * \return iterator into map at ptr or end() if not found.
   */
  Iterator find(Key ptr) noexcept;
  ConstIterator find(Key ptr) const noexcept;

  /*!
   * \brief Iterator to first value or end() if empty.
   */
  ConstIterator begin() const;
  Iterator begin();

  /*!
   * \brief Iterator to one-past-last value.
   */
  ConstIterator end() const;
  Iterator end();

  /*!
   * \brief Remove an entry from the map.
   */
  void erase(Key ptr);
  void erase(Iterator iter);
  void erase(ConstIterator oter);

  /*!
   * \brief Remove/deallocate the last found entry.
   *
   * WARNING: Use this
   * with caution, only directly after using a method above.
   * erase(Key) is safer, but requires an additional lookup.
   */
  void removeLast();

  /*!
   * \brief Clear all entris from the map.
   */
  void clear() noexcept;

  /*!
   * \brief Return number of entries in the map
   */
  std::size_t size() const noexcept;

 private:
  // Helper method for public findOrBefore()
  Key doFindOrBefore(Key ptr) const noexcept;

  // Helper method for insertion
  template <typename... Args>
  std::pair<Iterator, bool> doInsert(Key ptr, Args&&... args) noexcept;

  mutable Judy* m_array;    // Judy array
  mutable JudySlot* m_last; // last found value in judy array
  mutable uintptr_t m_oper; // address of last object to set internal judy state
  FixedMallocPool m_pool;   // value pool
  std::size_t m_size;       // number of objects stored
};

} // end of namespace util
} // end of namespace umpire

#include "umpire/util/MemoryMap.inl"

#endif // UMPIRE_MemoryMap_HPP
