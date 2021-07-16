//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/detail/fixed_malloc_pool.hpp"
#include "umpire/detail/log.hpp"
#include "umpire/detail/macros.hpp"

#include "umpire/tpl/judy/judy.h"

#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>
#include <mutex>

namespace umpire {
namespace detail {

namespace {

// Judy: number of Integers in a key
static constexpr unsigned int judy_depth{1};

// Judy: max height of stack
static constexpr unsigned int judy_max_levels{sizeof(uintptr_t)};

// Judy: length of key in bytes
static constexpr unsigned int judy_max{judy_depth * JUDY_key_size};

} // end anonymous namespace


// Tags for iterator constructors
struct iterator_begin {};
struct iterator_end {};

/*!
 * \brief A fast replacement for std::map<void*,Value> for a generic Value.
 *
 * This uses FixedMallocPool and Judy arrays and provides forward
 * const and non-const iterators.
 */
template <typename V>
class memory_map
{
public:
  using Key = void*;
  using Value = V;
  using KeyValuePair = std::pair<Key, Value*>;

  template <bool Const = false>
  class Iterator_
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Value;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    using Map = typename std::conditional<Const, const memory_map<Value>, memory_map<Value>>::type;
    using ValuePtr = typename std::conditional<Const, const Value*, Value*>::type;

    using Content = std::pair<Key, ValuePtr>;
    using Reference = typename std::conditional<Const, const Content&, Content&>::type;
    using Pointer = typename std::conditional<Const, const Content*, Content*>::type;

    Iterator_(Map* map, Key ptr) :
      m_map{map}, 
      m_pair{std::make_pair(ptr, m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr)}
    {}

    Iterator_(Map* map, iterator_begin) :
      m_map{map}, 
      m_pair{}
    {
      m_pair.first = nullptr;
      m_map->m_last = judy_strt(m_map->m_array, reinterpret_cast<const unsigned char*>(&m_pair.first), 0);
      judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first), judy_max);
      m_map->m_oper = reinterpret_cast<uintptr_t>(this);
      m_pair.second = m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr;
    }

    Iterator_(Map* map, iterator_end) :
      m_map{map}, 
      m_pair{std::make_pair(nullptr, static_cast<ValuePtr>(nullptr))}
    {}

    template<bool OtherConst>
    Iterator_(const Iterator_<OtherConst>& other) :
      m_map{other.m_map}, 
      m_pair{other.m_pair}
    {}
  

    Reference operator*()
    {
      return m_pair;
    }

    Pointer operator->()
    {
      return &m_pair;
    }

    Iterator_& operator++()
    {
      // Check whether this object was not the last to set the internal judy state
      if (m_pair.first && m_map->m_oper != reinterpret_cast<uintptr_t>(this)) {
        // Seek m_array internal position
        judy_slot(m_map->m_array, reinterpret_cast<const unsigned char*>(&m_pair.first), judy_max);
      }
      m_map->m_last = judy_nxt(m_map->m_array);
      m_map->m_oper = reinterpret_cast<uintptr_t>(this);

      if (!m_map->m_last) {
        // Reached end
        m_pair.first = nullptr;
      }
      else {
        // Update m_last and pair
        judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first), judy_max);
        m_pair.second = reinterpret_cast<ValuePtr>(*m_map->m_last);
      }

      return *this;
    }

    Iterator_ operator++(int)
    {
      Iterator tmp{*this};
      ++(*this);
      return tmp;
    }

    template <bool OtherConst>
    bool operator==(const Iterator_<OtherConst>& other) const
    {
      return m_map == other.m_map && m_pair.first == other.m_pair.first;
    }

    template <bool OtherConst>
    bool operator!=(const Iterator_<OtherConst>& other) const
    {
      return !(*this == other);
    }

  private:
    Map* m_map;
    Content m_pair;
  };

  template <bool Const> friend class Iterator_;

  using Iterator = Iterator_<false>;
  using ConstIterator = Iterator_<true>;

  memory_map():
    m_array{nullptr},
    m_last{nullptr},
    m_oper{0},
    m_pool{sizeof(Value)},
    m_size{0}
  {
    m_array = judy_open(judy_max_levels, judy_depth);
  }

  ~memory_map()
  {
    clear();
    judy_close(m_array);
  }

  // Would require a deep copy of the Judy data
  memory_map(const memory_map&) = delete;

  /*!
   * \brief Insert Value at ptr in the map if ptr does not exist. Uses
   * copy constructor on Value once.
   *
   * \return Pair of iterator position into map and boolean value
   * whether entry was added. The iterator will be set to end() if no
   * insertion was made.
   */
  std::pair<Iterator, bool> insert(Key ptr, const Value& val) noexcept
  {
    UMPIRE_LOG(Debug, "ptr = " << ptr);

    auto it_pair = doInsert(ptr, val);
    it_pair.second = !it_pair.second;

    return it_pair;
  }

  /*!
   * \brief Insert a key-value pair if pair.first does not exist as a
   * key. Must have first and second fields. Calls the first version.
   *
   * \return See alternative version.
   */
  template<typename P>
  std::pair<Iterator, bool> insert(P&& pair) noexcept
  {
    return insert(pair.first, pair.second);
  }

  /*!
   * \brief Emplaces a new value at ptr in the map, forwarding args to
   * the placement new constructor.
   *
   * \return See alternative version.
   */
  template <typename... Args>
  std::pair<Iterator, bool> insert(Key ptr, Args&&... args) noexcept
  {
  UMPIRE_LOG(Debug, "ptr = " << ptr);
  return doInsert(ptr, std::forward<Args>(args)...);
  }

  /*!
   * \brief Find a value at ptr.
   *
   * \return iterator into map at ptr or preceeding position.
   */
  Iterator findOrBefore(Key ptr) noexcept
  {
    ptr = doFindOrBefore(ptr);
    return Iterator{this, ptr};
  }

  ConstIterator findOrBefore(Key ptr) const noexcept
  {
    ptr = doFindOrBefore(ptr);
    return ConstIterator{this, ptr};
  }

  /*!
   * \brief Find a value at ptr.
   *
   * \return iterator into map at ptr or end() if not found.
   */
  Iterator find(Key ptr) noexcept
  {
    UMPIRE_LOG(Debug, "ptr = " << ptr);

    // Find the ptr and update m_oper
    m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
    m_oper = reinterpret_cast<uintptr_t>(this);
    return m_last ? Iterator{this, ptr} : Iterator{this, iterator_end{}};
  }

  ConstIterator find(Key ptr) const noexcept
  {
    UMPIRE_LOG(Debug, "ptr = " << ptr);

    // Find the ptr and update m_oper
    m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
    m_oper = reinterpret_cast<uintptr_t>(this);
    return m_last ? ConstIterator{this, ptr} : ConstIterator{this, iterator_end{}};
  }

  /*!
   * \brief Iterator to first value or end() if empty.
   */
  ConstIterator begin() const
  {
    return ConstIterator{this, iterator_begin{}};
  }

  Iterator begin()
  {
    return Iterator{this, iterator_begin{}};
  }

  /*!
   * \brief Iterator to one-past-last value.
   */
  ConstIterator end() const
  {
    return ConstIterator{this, iterator_end{}};

  }

  Iterator end()
  {
    return Iterator{this, iterator_end{}};
  }

  /*!
   * \brief Remove an entry from the map.
   */
  void erase(Key ptr)
  {
    UMPIRE_LOG(Debug, "ptr = " << ptr);

    // Locate ptr and update m_oper
    m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
    m_oper = reinterpret_cast<uintptr_t>(this);

    // If found, remove it
    if (m_last) {
      removeLast();
    } else {
      UMPIRE_ERROR("Could not remove ptr = " << ptr);
    }
  }

  void erase(Iterator iter)
  {
    erase(iter->first);
  }

  void erase(ConstIterator iter)
  {
    erase(iter->first);

  }

  /*!
   * \brief Remove/deallocate the last found entry.
   *
   * WARNING: Use this
   * with caution, only directly after using a method above.
   * erase(Key) is safer, but requires an additional lookup.
   */
  void removeLast()
  {
    auto v = reinterpret_cast<Value*>(*m_last);
    UMPIRE_ASSERT(v != nullptr);

    UMPIRE_LOG(Debug, "value pointer = " << v);

    // Manually call destructor
    v->~Value();

    // Mark as deallocated in the pool
    m_pool.deallocate(v);

    // Delete the key and cell for the current stack entry
    judy_del(m_array);

    // Decrease size
    --m_size;
  }

  /*!
   * \brief Clear all entris from the map.
   */
  void clear() noexcept
  {
    // Loop over the level 0 tree
    Key key{0};
    for(m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0);
        m_last;
        m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0))
      removeLast();

    m_size = 0;
  }

  /*!
   * \brief Return number of entries in the map
   */
  std::size_t size() const noexcept
  {
    return m_size;
  }

private:
  // Helper method for public findOrBefore()
  Key doFindOrBefore(Key ptr) const noexcept
  {
    UMPIRE_LOG(Debug, "ptr = " << ptr);

    // Find the ptr and update m_oper
    m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
    m_oper = reinterpret_cast<uintptr_t>(this);

    Key parent_ptr{0};
    judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);

    const Value* value{m_last ? reinterpret_cast<const Value*>(*m_last) : nullptr};

    // If the ptrs do not match, or the key does not exist, get the previous entry
    if (parent_ptr != ptr || !value)
    {
      m_last = judy_prv(m_array);
      judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);
    }
    UMPIRE_LOG(Debug, "returning " << parent_ptr);

    return parent_ptr;
  }

  // Helper method for insertion
  template <typename... Args>
  std::pair<Iterator, bool> doInsert(Key ptr, Args&&... args) noexcept
  {
    // Find the ptr and update m_oper
    m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
    m_oper = reinterpret_cast<uintptr_t>(this);
    UMPIRE_ASSERT(m_last);

    auto pval = reinterpret_cast<Value**>(m_last);
    const bool not_found{!(*pval)};

    if (not_found) {
      // Create it and increment size
      (*pval) = new (m_pool.allocate()) Value{std::forward<Args>(args)...};
      ++m_size;
    }

    return std::make_pair(Iterator{this, ptr}, not_found);
  }

  mutable Judy* m_array;    // Judy array
  mutable JudySlot* m_last; // last found value in judy array
  mutable uintptr_t m_oper; // address of last object to set internal judy state
  fixed_malloc_pool m_pool;   // value pool
  std::size_t m_size;            // number of objects stored
};


} // end of namespace util
} // end of namespace umpire
