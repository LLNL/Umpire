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
#ifndef UMPIRE_MemoryMap_INL
#define UMPIRE_MemoryMap_INL

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

namespace {

// Judy: number of Integers in a key
static constexpr unsigned int judy_depth = 1;

// Judy: max height of stack
static constexpr unsigned int judy_max_levels = sizeof(uintptr_t);

// Judy: length of key in bytes
static constexpr unsigned int judy_max = judy_depth * JUDY_key_size;

} // end anonymous namespace

// MemoryMap
template <typename V>
MemoryMap<V>::MemoryMap() :
  m_array{nullptr},
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
std::pair<typename MemoryMap<V>::Iterator, bool>
MemoryMap<V>::get(void* ptr, Args&&... args) noexcept
{
  // Find the ptr and update m_oper
  m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  UMPIRE_ASSERT(m_last);

  auto pval = reinterpret_cast<Value**>(m_last);

  const bool found = (*pval != nullptr);

  if (!found) {
    // Create it and increment size
    (*pval) = new (m_pool.allocate()) Value{args...};
    ++m_size;
  }

  return std::make_pair(Iterator{this}, found);
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::insert(void* ptr, const Value& val)
{
  // Insert the ptr (cell) and update m_oper
  m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  UMPIRE_ASSERT(m_last);

  auto pval{reinterpret_cast<Value**>(m_last)};

  // There should not already be a record here
  if (*pval) {
    UMPIRE_ERROR("Trying to insert at" << ptr << "but already exists");
  }

  // Create it
  (*pval) = new (m_pool.allocate()) Value{val};

  // Increment size
  ++m_size;

  return Iterator{this};
}

template <typename V>
void MemoryMap<V>::doFindOrBefore(void* ptr) const noexcept
{
  // Find the ptr and update m_oper
  m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);

  Key parent_ptr = 0;
  judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);

  // If the ptrs do not match, or the key does not exist, get the previous entry
  if (parent_ptr != ptr || !m_last)
  {
    m_last = judy_prv(m_array);
  }
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::findOrBefore(void* ptr) noexcept
{
  doFindOrBefore(ptr);
  return Iterator{this};
}

template <typename V>
typename MemoryMap<V>::ConstIterator
MemoryMap<V>::findOrBefore(void* ptr) const noexcept
{
  doFindOrBefore(ptr);
  return ConstIterator{this};
}

template <typename V>
typename MemoryMap<V>::Iterator MemoryMap<V>::find(void* ptr) noexcept
{
  // Find the ptr and update m_oper
  m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  return m_last ? Iterator{this} : Iterator{this, iterator_end{}};
}

template <typename V>
typename MemoryMap<V>::ConstIterator
MemoryMap<V>::find(void* ptr) const noexcept
{
  // Find the ptr and update m_oper
  m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  m_oper = reinterpret_cast<uintptr_t>(this);
  return m_last ? ConstIterator{this} : ConstIterator{this, iterator_end{}};
}

template <typename V>
void
MemoryMap<V>::remove(void* ptr)
{
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
void MemoryMap<V>::clear()
{
  // Loop over the level 0 tree
  Key key{0};
  while((m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0)) != nullptr)
    removeLast();
}

template <typename V>
size_t MemoryMap<V>::size() const noexcept
{
  return m_size;
}

// Iterator
template <typename V>
void MemoryMap<V>::removeLast()
{
  auto v{reinterpret_cast<Value*>(*m_last)};

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
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map) :
  m_map{map}, m_pair{}
{
  judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first), judy_max);
  m_pair.second = m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr;
  m_map->m_oper = reinterpret_cast<uintptr_t>(this);
}

template <typename V>
template <bool Const>
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map, iterator_begin) :
  m_map{map}, m_pair{}
{
  m_map->m_last = judy_strt(m_map->m_array, reinterpret_cast<const unsigned char*>(&m_pair.first), judy_max);
  judy_key(m_map->m_array, reinterpret_cast<unsigned char*>(&m_pair.first), judy_max);
  m_map->m_oper = reinterpret_cast<uintptr_t>(this);
  m_pair.second = m_map->m_last ? reinterpret_cast<ValuePtr>(*m_map->m_last) : nullptr;
}

template <typename V>
template <bool Const>
MemoryMap<V>::Iterator_<Const>::Iterator_(Map* map, iterator_end) :
  m_map{map}, m_pair{std::make_pair(nullptr, static_cast<ValuePtr>(nullptr))}
{
}

template <typename V>
template <bool Const>
template <bool OtherConst>
MemoryMap<V>::Iterator_<Const>::Iterator_(const Iterator_<OtherConst>& other) :
  m_map{other.m_map}, m_pair{other.m_pair}
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
    judy_slot(m_map->m_array, reinterpret_cast<const unsigned char*>(&m_pair.first), judy_max);
  }
  m_map->m_last = judy_nxt(m_map->m_array);
  m_map->m_oper = true;

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
bool MemoryMap<V>::Iterator_<Const>::operator==(const MemoryMap<V>::Iterator_<OtherConst>& other) const
{
  return m_map == other.m_map && m_pair.first == other.m_pair.first;
}

template <typename V>
template <bool Const>
template <bool OtherConst>
bool MemoryMap<V>::Iterator_<Const>::operator!=(const MemoryMap<V>::Iterator_<OtherConst>& other) const
{
  return !(*this == other);
}

template <typename V>
typename MemoryMap<V>::ConstIterator
MemoryMap<V>::begin() const
{
  return ConstIterator{this, iterator_begin{}};
}

template <typename V>
typename MemoryMap<V>::Iterator
MemoryMap<V>::begin()
{
  return Iterator{this, iterator_begin{}};
}

template <typename V>
typename MemoryMap<V>::ConstIterator
MemoryMap<V>::end() const
{
  return ConstIterator{this, iterator_end{}};
}

template <typename V>
typename MemoryMap<V>::Iterator
MemoryMap<V>::end()
{
  return Iterator{this, iterator_end{}};
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MemoryMap_INL
