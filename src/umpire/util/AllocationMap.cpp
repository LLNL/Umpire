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
#include "umpire/util/AllocationMap.hpp"

#include "umpire/util/FixedMallocPool.hpp"

#include "umpire/util/Macros.hpp"

#include <sstream>
#include <type_traits>

namespace umpire {
namespace util {

template <typename T>
struct ListBlock
{
  T rec;
  ListBlock* prev;
};

class RecordList
{
public:
  using BlockType = ListBlock<AllocationRecord>;

  // Iterator for RecordList
  template <bool Const = false>
  class Iterator : public std::iterator<std::forward_iterator_tag, AllocationRecord>
  {
  public:
    using Value = AllocationRecord;
    using Reference = typename std::conditional<Const, Value const&, Value&>::type;
    using Pointer = typename std::conditional<Const, Value const*, Value*>::type;

    Iterator(const RecordList* list, bool end);
    Iterator(const Iterator&) = default;

    Reference operator*();
    Pointer operator->();
    Iterator& operator++();
    Iterator operator++(int);

    template <bool OtherConst>
    bool operator==(const Iterator<OtherConst>& other);

    template <bool OtherConst>
    bool operator!=(const Iterator<OtherConst>& other);
  private:
    const RecordList *m_list;
    RecordList::BlockType* m_curr;
  };

  // Iterator needs access to m_tail
  template <bool> friend class Iterator;

  RecordList(AllocationRecord record);
  ~RecordList();

  void push_back(const AllocationRecord& rec);
  AllocationRecord pop_back();

  Iterator<true> begin() const;
  Iterator<true> end() const;

  size_t size() const;
  bool empty() const;
  AllocationRecord* back();
  const AllocationRecord* back() const;

private:
  BlockType* m_tail;
  size_t m_length;
};

static umpire::util::FixedMallocPool block_pool(sizeof(ListBlock<AllocationRecord>));

// Record List
RecordList::RecordList(AllocationRecord record)
  : m_tail(nullptr), m_length(0)
{
  push_back(record);
}

RecordList::~RecordList()
{
  BlockType* curr = m_tail;
  while (curr) {
    BlockType* prev = curr->prev;
    block_pool.deallocate(curr);
    curr = prev;
  }
}

void RecordList::push_back(const AllocationRecord& rec)
{
  BlockType* curr = static_cast<BlockType*>(block_pool.allocate());
  curr->prev = m_tail;
  curr->rec = rec;
  m_tail = curr;
  m_length++;
}

AllocationRecord RecordList::pop_back()
{
  AllocationRecord ret{};
  if (m_length > 0) {
    ret = m_tail->rec;
    BlockType* prev = m_tail->prev;
    block_pool.deallocate(m_tail);
    m_tail = prev;
    m_length--;
  }
  return ret;
}

RecordList::Iterator<true> RecordList::begin() const
{
  return RecordList::Iterator<true>{this, false};
}

RecordList::Iterator<true> RecordList::end() const
{
  return RecordList::Iterator<true>{this, true};
}

size_t RecordList::size() const { return m_length; }
bool RecordList::empty() const { return size() == 0; }

AllocationRecord* RecordList::back() { return &m_tail->rec; }

const AllocationRecord* RecordList::back() const { return &m_tail->rec; }

// RecordList::Iterator
template <bool Const>
RecordList::Iterator<Const>::Iterator(const RecordList* list, bool end)
  : m_list(list), m_curr(end ? nullptr : m_list->m_tail)
{
}

template <bool Const>
typename RecordList::Iterator<Const>::Reference
RecordList::Iterator<Const>::operator*()
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  return m_curr->rec;
}

template <bool Const>
typename RecordList::Iterator<Const>::Pointer
RecordList::Iterator<Const>::operator->()
{
  return m_curr ? &(m_curr->rec) : nullptr;
}

template <bool Const>
RecordList::Iterator<Const>& RecordList::Iterator<Const>::operator++()
{
  m_curr = m_curr->prev;
  return *this;
}

template <bool Const>
RecordList::Iterator<Const> RecordList::Iterator<Const>::operator++(int)
{
  Iterator<Const> tmp{*this};
  m_curr = m_curr->prev;
  return tmp;
}

template <bool Const>
template <bool OtherConst>
bool RecordList::Iterator<Const>::operator==(const RecordList::Iterator<OtherConst>& other)
{
  return m_list == other.m_list && m_curr == other.m_curr;
}

template <bool Const>
template <bool OtherConst>
bool RecordList::Iterator<Const>::operator!=(const RecordList::Iterator<OtherConst>& other)
{
  return !(*this == other);
}

// AllocationMap
AllocationMap::AllocationMap() :
  m_map(new MapType{}), m_size(0), m_mutex()
{
}

AllocationMap::~AllocationMap()
{
  delete m_map;
}

void AllocationMap::insert(void* ptr, AllocationRecord record)
{
  UMPIRE_LOCK;
  UMPIRE_LOG(Debug, "Inserting " << ptr);

  auto ret = m_map->get(ptr, record);
  if (ret.second) ret.first->push_back(record);

  UMPIRE_UNLOCK;
  ++m_size;
}

const AllocationRecord* AllocationMap::find(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);

  auto alloc_record = findRecord(ptr);

  if (alloc_record) {
    return alloc_record;
  } else {
#if !defined(NDEBUG)
    // use this from a debugger to dump the contents of the AllocationMap
    printAll();
#endif
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

AllocationRecord* AllocationMap::find(void* ptr)
{
  return const_cast<AllocationRecord*>(const_cast<const AllocationMap*>(this)->find(ptr));
}

const AllocationRecord* AllocationMap::findRecord(void* ptr) const noexcept
{
  const AllocationRecord* alloc_record{nullptr};

  UMPIRE_LOCK;

  auto iter = m_map->find(ptr);

  // If a list was found, return its tail
  if (iter != m_map->end()) {
    alloc_record = iter->back();

    const uintptr_t parent_ptr{reinterpret_cast<uintptr_t>(alloc_record->ptr)};

    if ((parent_ptr + alloc_record->size) > reinterpret_cast<uintptr_t>(ptr)) {
      UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr
                 << " with size " << alloc_record->size);
    }
  }

  UMPIRE_UNLOCK;

  return alloc_record;
}

AllocationRecord* AllocationMap::findRecord(void* ptr) noexcept
{
  return const_cast<AllocationRecord*>(const_cast<const AllocationMap*>(this)->findRecord(ptr));
}


AllocationRecord AllocationMap::remove(void* ptr)
{
  AllocationRecord ret{};

  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "Removing " << ptr);

    auto iter = m_map->find(ptr);
    if (iter != m_map->end()) {
      ret = iter->pop_back();
      if (iter->empty()) m_map->removeLast();
    }
    else {
      UMPIRE_ERROR("Cannot remove " << ptr);
    }

    --m_size;

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return ret;
}

bool AllocationMap::contains(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
  return (findRecord(ptr) != nullptr);
}

void AllocationMap::clear()
{
  m_map->clear();
}

size_t AllocationMap::size() const { return m_size; }

void
AllocationMap::print(const std::function<bool (const AllocationRecord&)>&& UMPIRE_UNUSED_ARG(pred),
                     std::ostream& UMPIRE_UNUSED_ARG(os)) const
{
  // TODO TBD
  // uintptr_t key = 0;
  // for(m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0);
  //     m_last != nullptr;
  //     m_last = judy_nxt(m_array)) {
  //   auto list = reinterpret_cast<RecordList*>(*m_last);

  //   void* addr;
  //   judy_key(m_array, reinterpret_cast<unsigned char*>(&addr), judy_max);

  //   std::stringstream ss;
  //   bool any_match = false;
  //   ss << addr << " {" << std::endl;
  //   for (auto iter{list->begin()}; iter != list->end(); ++iter) {
  //     if (pred(*iter)) {
  //       any_match = true;
  //       ss << iter->size <<
  //         " [ " << reinterpret_cast<void*>(iter->ptr) <<
  //         " -- " << reinterpret_cast<void*>(static_cast<unsigned char*>(iter->ptr)+iter->size) <<
  //         " ] " << std::endl;
  //     }
  //   }
  //   ss << "}" << std::endl;

  //   if (any_match) {
  //     os << ss.str();
  //   }
  // }
}

void AllocationMap::printAll(std::ostream& os) const
{
  os << "🔍 Printing allocation map contents..." << std::endl;

  print([] (const AllocationRecord&) { return true; }, os);

  os << "done." << std::endl;
}


AllocationMap::MapType::template Iterator<true> AllocationMap::begin() const
{
  return static_cast<const MapType*>(m_map)->begin();
}

AllocationMap::MapType::template Iterator<true> AllocationMap::end() const
{
  return static_cast<const MapType*>(m_map)->end();
}

} // end of namespace util
} // end of namespace umpire
