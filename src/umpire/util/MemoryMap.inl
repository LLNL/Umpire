//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryMap_INL
#define UMPIRE_MemoryMap_INL

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

namespace {

// Judy: number of Integers in a key
static constexpr unsigned int judy_depth{1};

// Judy: max height of stack
static constexpr unsigned int judy_max_levels{sizeof(uintptr_t)};

// Judy: length of key in bytes
static constexpr unsigned int judy_max{judy_depth * JUDY_key_size};

} // end anonymous namespace

// MemoryMap
template <typename V>
MemoryMap<V>::MemoryMap()
    : m_array{nullptr},
      m_last{nullptr},
      m_oper{0},
      m_pool{sizeof(Value)},
      m_size{0}
{
  // Create new judy array
  m_array = judy_open(judy_max_levels, judy_depth);
}

template <typename V>
MemoryMap<V>::~MemoryMap()
{
  // Delete all entries
  clear();

  // Close the judy array, freeing all memory
  judy_close(m_array);
}

template <typename V>
template <typename... Args>
std::pair<typename MemoryMap<V>::Iterator, bool> MemoryMap<V>::doInsert(
    Key ptr, Args&&... args) noexcept
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

template <typename V>
std::pair<typename MemoryMap<V>::Iterator, bool> MemoryMap<V>::insert(
    Key ptr, const Value& val) noexcept
{
  UMPIRE_LOG(Debug, "ptr = " << ptr);

  auto it_pair = doInsert(ptr, val);
  it_pair.second = !it_pair.second;

  return it_pair;
}

template <typename V>
template <typename P>
std::pair<typename MemoryMap<V>::Iterator, bool> MemoryMap<V>::insert(
    P&& pair) noexcept
{
  return insert(pair.first, pair.second);
}

template <typename V>
template <typename... Args>
std::pair<typename MemoryMap<V>::Iterator, bool> MemoryMap<V>::insert(
    Key ptr, Args&&... args) noexcept
{
  UMPIRE_LOG(Debug, "ptr = " << ptr);
  return doInsert(ptr, std::forward<Args>(args)...);
}

template <typename V>
typename MemoryMap<V>::Key MemoryMap<V>::doFindOrBefore(Key ptr) const noexcept
{
  UMPIRE_LOG(Debug, "ptr = " << ptr);

  // Find the ptr and update m_oper
  m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);

  Key parent_ptr{0};
  judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);

  const Value* value{m_last ? reinterpret_cast<const Value*>(*m_last)
                            : nullptr};

  // If the ptrs do not match, or the key does not exist, get the previous entry
  if (parent_ptr != ptr || !value) {
    m_last = judy_prv(m_array);
    judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);
  }
  UMPIRE_LOG(Debug, "returning " << parent_ptr);

  return parent_ptr;
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::findOrBefore(Key ptr) noexcept
{
  ptr = doFindOrBefore(ptr);
  return Iterator{this, ptr};
}

template <typename V>
typename MemoryMap<V>::ConstIterator MemoryMap<V>::findOrBefore(Key ptr) const
    noexcept
{
  ptr = doFindOrBefore(ptr);
  return ConstIterator{this, ptr};
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::find(Key ptr) noexcept
{
  UMPIRE_LOG(Debug, "ptr = " << ptr);

  // Find the ptr and update m_oper
  m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  return m_last ? Iterator{this, ptr} : Iterator{this, iterator_end{}};
}

template <typename V>
typename MemoryMap<V>::ConstIterator MemoryMap<V>::find(Key ptr) const noexcept
{
  UMPIRE_LOG(Debug, "ptr = " << ptr);

  // Find the ptr and update m_oper
  m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  return m_last ? ConstIterator{this, ptr}
                : ConstIterator{this, iterator_end{}};
}

template <typename V>
void MemoryMap<V>::erase(Key ptr)
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

template <typename V>
void MemoryMap<V>::erase(Iterator iter)
{
  erase(iter->first);
}

template <typename V>
void MemoryMap<V>::erase(ConstIterator iter)
{
  erase(iter->first);
}

template <typename V>
void MemoryMap<V>::clear() noexcept
{
  // Loop over the level 0 tree
  Key key{0};
  for (m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0);
       m_last;
       m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0))
    removeLast();

  m_size = 0;
}

template <typename V>
std::size_t MemoryMap<V>::size() const noexcept
{
  return m_size;
}

template <typename V>
void MemoryMap<V>::removeLast()
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

template <typename V>
template <bool Const>
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map, Key ptr)
    : m_map{map},
      m_pair{std::make_pair(
          ptr,
          m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr)}
{
}

template <typename V>
template <bool Const>
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map, iterator_begin)
    : m_map{map}, m_pair{}
{
  m_pair.first = nullptr;
  m_map->m_last = judy_strt(
      m_map->m_array, reinterpret_cast<const unsigned char*>(&m_pair.first), 0);
  judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first),
           judy_max);
  m_map->m_oper = reinterpret_cast<uintptr_t>(this);
  m_pair.second =
      m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr;
}

template <typename V>
template <bool Const>
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map, iterator_end)
    : m_map{map},
      m_pair{std::make_pair(nullptr, static_cast<ValuePtr>(nullptr))}
{
}

template <typename V>
template <bool Const>
template <bool OtherConst>
MemoryMap<V>::Iterator_<Const>::Iterator_(const Iterator_<OtherConst>& other)
    : m_map{other.m_map}, m_pair{other.m_pair}
{
}

template <typename V>
template <bool Const>
typename MemoryMap<V>::template Iterator_<Const>::Reference
    MemoryMap<V>::Iterator_<Const>::operator*()
{
  return m_pair;
}

template <typename V>
template <bool Const>
typename MemoryMap<V>::template Iterator_<Const>::Pointer
    MemoryMap<V>::Iterator_<Const>::operator->()
{
  return &m_pair;
}

template <typename V>
template <bool Const>
typename MemoryMap<V>::template Iterator_<Const>&
MemoryMap<V>::Iterator_<Const>::operator++()
{
  // Check whether this object was not the last to set the internal judy state
  if (m_pair.first && m_map->m_oper != reinterpret_cast<uintptr_t>(this)) {
    // Seek m_array internal position
    judy_slot(m_map->m_array,
              reinterpret_cast<const unsigned char*>(&m_pair.first), judy_max);
  }
  m_map->m_last = judy_nxt(m_map->m_array);
  m_map->m_oper = reinterpret_cast<uintptr_t>(this);

  if (!m_map->m_last) {
    // Reached end
    m_pair.first = nullptr;
  } else {
    // Update m_last and pair
    judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first),
             judy_max);
    m_pair.second = reinterpret_cast<ValuePtr>(*m_map->m_last);
  }

  return *this;
}

template <typename V>
template <bool Const>
typename MemoryMap<V>::template Iterator_<Const>
MemoryMap<V>::Iterator_<Const>::operator++(int)
{
  Iterator tmp{*this};
  ++(*this);
  return tmp;
}

template <typename V>
template <bool Const>
template <bool OtherConst>
bool MemoryMap<V>::Iterator_<Const>::operator==(
    const MemoryMap<V>::Iterator_<OtherConst>& other) const
{
  return m_map == other.m_map && m_pair.first == other.m_pair.first;
}

template <typename V>
template <bool Const>
template <bool OtherConst>
bool MemoryMap<V>::Iterator_<Const>::operator!=(
    const MemoryMap<V>::Iterator_<OtherConst>& other) const
{
  return !(*this == other);
}

template <typename V>
typename MemoryMap<V>::ConstIterator MemoryMap<V>::begin() const
{
  return ConstIterator{this, iterator_begin{}};
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::begin()
{
  return Iterator{this, iterator_begin{}};
}

template <typename V>
typename MemoryMap<V>::ConstIterator MemoryMap<V>::end() const
{
  return ConstIterator{this, iterator_end{}};
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::end()
{
  return Iterator{this, iterator_end{}};
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MemoryMap_INL
