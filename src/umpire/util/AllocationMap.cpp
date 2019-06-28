//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/AllocationMap.hpp"

#include "umpire/util/FixedMallocPool.hpp"

#include "umpire/util/Macros.hpp"

#include <sstream>
#include <type_traits>
#include <utility>

namespace umpire {
namespace util {

static umpire::util::FixedMallocPool block_pool(sizeof(RecordList::Block<AllocationRecord>));

// Record List
RecordList::RecordList(AllocationRecord record)
  : m_tail{nullptr}, m_length{0}
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
  if (m_length == 0) {
    UMPIRE_ERROR("pop_back() called, but m_length == 0");
  }

  const AllocationRecord ret = m_tail->rec;
  BlockType* prev = m_tail->prev;

  // Deallocate and move tail pointer
  block_pool.deallocate(m_tail);
  m_tail = prev;

  // Reduce size
  m_length--;

  return ret;
}

RecordList::ConstIterator RecordList::begin() const
{
  return RecordList::ConstIterator{this, iterator_begin{}};
}

RecordList::ConstIterator RecordList::end() const
{
  return RecordList::ConstIterator{this, iterator_end{}};
}

std::size_t RecordList::size() const { return m_length; }
bool RecordList::empty() const { return size() == 0; }
AllocationRecord* RecordList::back() { return &m_tail->rec; }
const AllocationRecord* RecordList::back() const { return &m_tail->rec; }

RecordList::ConstIterator::ConstIterator()
  : m_list(nullptr), m_curr(nullptr)
{
}

RecordList::ConstIterator::ConstIterator(const RecordList* list, iterator_begin)
  : m_list(list), m_curr(m_list->m_tail)
{
}

RecordList::ConstIterator::ConstIterator(const RecordList* list, iterator_end)
  : m_list(list), m_curr(nullptr)
{
}

const AllocationRecord&
RecordList::ConstIterator::operator*()
{
  return *operator->();
}

const AllocationRecord*
RecordList::ConstIterator::operator->()
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  return &m_curr->rec;
}

RecordList::ConstIterator& RecordList::ConstIterator::operator++()
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  m_curr = m_curr->prev;
  return *this;
}

RecordList::ConstIterator RecordList::ConstIterator::operator++(int)
{
  ConstIterator tmp{*this};
  this->operator++();
  return tmp;
}

bool RecordList::ConstIterator::operator==(const RecordList::ConstIterator& other) const
{
  return m_list == other.m_list && m_curr == other.m_curr;
}

bool RecordList::ConstIterator::operator!=(const RecordList::ConstIterator& other) const
{
  return !(*this == other);
}

// AllocationMap
AllocationMap::AllocationMap() :
  m_map{}, m_size{0}, m_mutex{}
{
}

void AllocationMap::insert(void* ptr, AllocationRecord record)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Inserting " << ptr);

  Map::Iterator iter{m_map.end()};
  bool found;

  std::tie(iter, found) = m_map.get(ptr, record);
  if (found) {
    // Record was not added
    iter->second->push_back(record);
  }
  // else (found = false)
  // -> get() already added record to the end of the RecordList for ptr

  ++m_size;
}

const AllocationRecord* AllocationMap::find(void* ptr) const
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Searching for " << ptr);

  const AllocationRecord* alloc_record = doFindRecord(ptr);

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

const AllocationRecord* AllocationMap::doFindRecord(void* ptr) const noexcept
{
  const AllocationRecord* alloc_record = nullptr;

  Map::ConstIterator iter = m_map.findOrBefore(ptr);

  // faster, equivalent way of checking iter != m_map->end()
  if (iter->second) {
    auto candidate = iter->second->back();
    UMPIRE_ASSERT(candidate->ptr <= ptr);

    // Check if ptr is inside candidate's allocation
    const bool in_candidate =
      (static_cast<char*>(candidate->ptr) + candidate->size)
      > static_cast<char*>(ptr) || (candidate->ptr == ptr);

    if (in_candidate) {
      UMPIRE_LOG(Debug, "Found " << ptr << " at " << candidate->ptr
                 << " with size " << candidate->size);
      alloc_record = candidate;
    }
    else {
      alloc_record = nullptr;
    }
  }

  return alloc_record;
}

const AllocationRecord* AllocationMap::findRecord(void* ptr) const noexcept
{
  std::lock_guard<std::mutex> lock(m_mutex);

  // Call method
  return doFindRecord(ptr);
}

AllocationRecord* AllocationMap::findRecord(void* ptr) noexcept
{
  return const_cast<AllocationRecord*>(const_cast<const AllocationMap*>(this)->findRecord(ptr));
}


AllocationRecord AllocationMap::remove(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  AllocationRecord ret;

  UMPIRE_LOG(Debug, "Removing " << ptr);

  auto iter = m_map.find(ptr);

  if (iter->second) {
    // faster, equivalent way of checking iter != m_map->end()
    ret = iter->second->pop_back();
    if (iter->second->empty()) m_map.removeLast();
  }
  else {
    UMPIRE_ERROR("Cannot remove " << ptr);
  }

  --m_size;

  return ret;
}

bool AllocationMap::contains(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
  return (findRecord(ptr) != nullptr);
}

void AllocationMap::clear()
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Clearing");

  m_map.clear();
  m_size = 0;
}

std::size_t AllocationMap::size() const
{
  return m_size;
}

void
AllocationMap::print(const std::function<bool (const AllocationRecord&)>&& pred,
                     std::ostream& os) const
{
  for (auto p : m_map) {
    std::stringstream ss;
    bool any_match = false;
    ss << p.first << " {" << std::endl;
    auto iter = p.second->begin();
    auto end = p.second->end();
    while (iter != end) {
      if (pred(*iter)) {
        any_match = true;
        ss << iter->size <<
          " [ " << reinterpret_cast<void*>(iter->ptr) <<
          " -- " << reinterpret_cast<void*>(static_cast<unsigned char*>(iter->ptr)+iter->size) <<
          " ] " << std::endl;
      }
      ++iter;
    }
    ss << "}" << std::endl;

    if (any_match) {
      os << ss.str();
    }
  }
}

void AllocationMap::printAll(std::ostream& os) const
{
  os << "🔍 Printing allocation map contents..." << std::endl;

  print([] (const AllocationRecord&) { return true; }, os);

  os << "done." << std::endl;
}


AllocationMap::ConstIterator AllocationMap::begin() const
{
  return AllocationMap::ConstIterator{this, iterator_begin{}};
}

AllocationMap::ConstIterator AllocationMap::end() const
{
  return AllocationMap::ConstIterator{this, iterator_end{}};
}

AllocationMap::ConstIterator::ConstIterator(const AllocationMap* map, iterator_begin) :
  m_outer_iter(map->m_map.begin()),
  m_inner_iter(m_outer_iter->first ? m_outer_iter->second->begin() : InnerIter{}),
  m_inner_end(m_outer_iter->first ? m_outer_iter->second->end() : InnerIter{}),
  m_outer_end(map->m_map.end())
{
}

AllocationMap::ConstIterator::ConstIterator(const AllocationMap* map, iterator_end) :
  m_outer_iter(map->m_map.end()),
  m_inner_iter(InnerIter{}),
  m_inner_end(InnerIter{}),
  m_outer_end(map->m_map.end())
{
}

const AllocationRecord&
AllocationMap::ConstIterator::operator*()
{
  return m_inner_iter.operator*();
}

const AllocationRecord*
AllocationMap::ConstIterator::operator->()
{
  return m_inner_iter.operator->();
}

AllocationMap::ConstIterator& AllocationMap::ConstIterator::operator++()
{
  ++m_inner_iter;
  if (m_inner_iter == m_inner_end) {
    ++m_outer_iter;
    if (m_outer_iter != m_outer_end) {
      m_inner_iter = m_outer_iter->second->begin();
      m_inner_end = m_outer_iter->second->end();
    } else {
      m_inner_iter = InnerIter{};
    }
  }
  return *this;
}

AllocationMap::ConstIterator AllocationMap::ConstIterator::operator++(int)
{
  ConstIterator tmp{*this};
  ++(*this);
  return tmp;
}

bool AllocationMap::ConstIterator::operator==(const AllocationMap::ConstIterator& other) const
{
  return m_outer_iter == other.m_outer_iter && m_inner_iter == other.m_inner_iter;
}

bool AllocationMap::ConstIterator::operator!=(const AllocationMap::ConstIterator& other) const
{
  return !(*this == other);
}

} // end of namespace util
} // end of namespace umpire
