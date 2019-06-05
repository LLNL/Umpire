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

} // anonymous namespace

template <typename Value>
MemoryMap<Value>::MemoryMap() :
  m_array{nullptr},
  m_last{nullptr},
  m_size{0},
  m_pool{sizeof(Value)}
{
  // Create new judy array
  m_array = judy_open(judy_max_levels, judy_depth);
}

template <typename Value>
MemoryMap<Value>::~MemoryMap()
{
  // Delete all entries
  clear();

  // Close the judy array, freeing all memory.
  judy_close(m_array);
}

template <typename Value>
template <typename... Args>
std::pair<typename MemoryMap<Value>::template Iterator<false>, bool> MemoryMap<Value>::get(void* ptr, Args&... args)
{
  // Find the key
  m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  UMPIRE_ASSERT(m_last);

  auto pval = reinterpret_cast<Value**>(m_last);

  const bool found{*pval != nullptr};

  if (!found) (*pval) = new (m_pool.allocate()) Value{args...};

  return std::make_pair(Iterator<false>{m_array, m_last, reinterpret_cast<KeyType>(ptr)}, found);
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<false> MemoryMap<Value>::insert(void* ptr, const Value& val)
{
  // Find the key
  m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  UMPIRE_ASSERT(m_last);

  auto pval = reinterpret_cast<Value**>(m_last);

  // There should not already be a record here
  if (*pval) {
    UMPIRE_ERROR("Trying to insert at" << ptr << "but already exists");
  }

  // Create it
  (*pval) = new (m_pool.allocate()) Value{val};

  return Iterator<false>{m_array, m_last, ptr};
}

template <typename Value>
typename MemoryMap<Value>::KeyType MemoryMap<Value>::doFind(void* ptr) const
{
  m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);

  KeyType parent_ptr{0};
  judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);

  // If the ptrs do not match, or the key does not exist, get the previous entry
  if (parent_ptr != reinterpret_cast<uintptr_t>(ptr) || !m_last)
  {
    m_last = judy_prv(m_array);
    // Find key associated to this one
    judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);
  }

  return parent_ptr;
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<false> MemoryMap<Value>::find(void* ptr)
{
  KeyType parent_ptr = doFind(ptr);
  return Iterator<false>{m_array, m_last, parent_ptr};
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<true>
MemoryMap<Value>::find(void* ptr) const
{
  KeyType parent_ptr = doFind(ptr);
  return Iterator<true>{m_array, m_last, parent_ptr};
}

template <typename Value>
void
MemoryMap<Value>::remove(void* ptr)
{
  // Locate ptr
  m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);

  // If found, remove it
  if (m_last && *m_last) removeLast();
}

template <typename Value>
void MemoryMap<Value>::clear()
{
  // Loop over the level 0 tree
  KeyType key{0};
  while((m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0)) != nullptr)
    removeLast();
}

template <typename Value>
void MemoryMap<Value>::removeLast()
{
  auto v = reinterpret_cast<Value*>(*m_last);

  // Manually call destructor
  v->~Value();

  // Mark as deallocated in the pool
  m_pool.deallocate(v);

  // Delete the key and cell for the current stack entry.
  judy_del(m_array);
}


template <typename Value>
template <bool Const>
typename MemoryMap<Value>::template Iterator<Const>::Content
MemoryMap<Value>::Iterator<Const>::makePair()
{
  return std::make_pair(reinterpret_cast<void*>(m_key), m_last ? reinterpret_cast<ValuePtr>(*m_last) : nullptr);
}

template <typename Value>
template <bool Const>
MemoryMap<Value>::Iterator<Const>::Iterator(Judy* array, JudySlot* last, MemoryMap<Value>::KeyType key) :
  m_array(array), m_last(last), m_key(key), m_pair(makePair())
{
}

template <typename Value>
template <bool Const>
MemoryMap<Value>::Iterator<Const>::Iterator(Judy* array, bool end) :
  m_array(array), m_last(nullptr), m_key(0), m_pair()
{
  if (!end) {
    m_last = judy_strt(m_array, reinterpret_cast<const unsigned char*>(&m_key), judy_max);
  } else {
    m_key = 0;
  }

  m_pair = makePair();
}

template <typename Value>
template <bool Const>
template <bool OtherConst>
MemoryMap<Value>::Iterator<Const>::Iterator(const Iterator<OtherConst>& other) :
  m_array(other.m_array), m_last(other.m_last), m_key(other.m_key), m_pair(makePair())
{
}

template <typename Value>
template <bool Const>
typename MemoryMap<Value>::template Iterator<Const>::Reference
MemoryMap<Value>::Iterator<Const>::operator*()
{
  return m_pair;
}

template <typename Value>
template <bool Const>
typename MemoryMap<Value>::template Iterator<Const>::Pointer
MemoryMap<Value>::Iterator<Const>::operator->()
{
  return &m_pair;
}

template <typename Value>
template <bool Const>
typename MemoryMap<Value>::template Iterator<Const>&
MemoryMap<Value>::Iterator<Const>::operator++()
{
  // Move to a new pointer
  auto new_slot = judy_strt(m_array, reinterpret_cast<const unsigned char*>(&m_key), judy_max);

  if (new_slot == m_last) {
    // Reached end
    m_key = 0;
  }
  else {
    // Update m_last
    m_last = new_slot;
  }

  // Update pair
  m_pair = makePair();

  return *this;
}

template <typename Value>
template <bool Const>
typename MemoryMap<Value>::template Iterator<Const>
MemoryMap<Value>::Iterator<Const>::operator++(int)
{
  Iterator tmp{*this};
  ++(*this);
  return tmp;
}

template <typename Value>
template <bool Const>
template <bool OtherConst>
bool MemoryMap<Value>::Iterator<Const>::operator==(const MemoryMap<Value>::Iterator<OtherConst>& other)
{
  return (m_array == other.m_array && m_key == other.m_key);
}

template <typename Value>
template <bool Const>
template <bool OtherConst>
bool MemoryMap<Value>::Iterator<Const>::operator!=(const MemoryMap<Value>::Iterator<OtherConst>& other)
{
  return !(*this == other);
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<true>
MemoryMap<Value>::begin() const
{
  return Iterator<true>{m_array, false};
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<false>
MemoryMap<Value>::begin()
{
  return Iterator<false>{m_array, false};
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<true>
MemoryMap<Value>::end() const
{
  return Iterator<true>{m_array, true};
}

template <typename Value>
typename MemoryMap<Value>::template Iterator<false>
MemoryMap<Value>::end()
{
  return Iterator<false>{m_array, true};
}


} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MemoryMap_INL
