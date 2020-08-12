//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/AllocationMap.hpp"

#include <functional>
#include <sstream>
#include <type_traits>
#include <utility>

#include "umpire/Replay.hpp"
#include "umpire/util/FixedMallocPool.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/backtrace.hpp"

namespace umpire {
namespace util {

// Record List
AllocationMap::RecordList::RecordList(AllocationMap& map,
                                      AllocationRecord record)
    : m_map{map}, m_tail{nullptr}, m_length{0}
{
  push_back(record);
}

AllocationMap::RecordList::~RecordList()
{
  RecordBlock* curr = m_tail;
  while (curr) {
    RecordBlock* prev = curr->prev;
    m_map.m_block_pool.deallocate(curr);
    curr = prev;
  }
}

void AllocationMap::RecordList::push_back(const AllocationRecord& rec)
{
  RecordBlock* curr = static_cast<RecordBlock*>(m_map.m_block_pool.allocate());
  curr->prev = m_tail;
  curr->rec = rec;
  m_tail = curr;
  m_length++;
}

AllocationRecord AllocationMap::RecordList::pop_back()
{
  if (m_length == 0) {
    UMPIRE_ERROR("pop_back() called, but m_length == 0");
  }

  const AllocationRecord ret = m_tail->rec;
  RecordBlock* prev = m_tail->prev;

  // Deallocate and move tail pointer
  m_map.m_block_pool.deallocate(m_tail);
  m_tail = prev;

  // Reduce size
  m_length--;

  return ret;
}

AllocationMap::RecordList::ConstIterator AllocationMap::RecordList::begin()
    const
{
  return AllocationMap::RecordList::ConstIterator{this, iterator_begin{}};
}

AllocationMap::RecordList::ConstIterator AllocationMap::RecordList::end() const
{
  return AllocationMap::RecordList::ConstIterator{this, iterator_end{}};
}

std::size_t AllocationMap::RecordList::size() const
{
  return m_length;
}

bool AllocationMap::RecordList::empty() const
{
  return size() == 0;
}

AllocationRecord* AllocationMap::RecordList::back()
{
  return &m_tail->rec;
}

const AllocationRecord* AllocationMap::RecordList::back() const
{
  return &m_tail->rec;
}

AllocationMap::RecordList::ConstIterator::ConstIterator()
    : m_list(nullptr), m_curr(nullptr)
{
}

AllocationMap::RecordList::ConstIterator::ConstIterator(const RecordList* list,
                                                        iterator_begin)
    : m_list(list), m_curr(m_list->m_tail)
{
}

AllocationMap::RecordList::ConstIterator::ConstIterator(const RecordList* list,
                                                        iterator_end)
    : m_list(list), m_curr(nullptr)
{
}

const AllocationRecord& AllocationMap::RecordList::ConstIterator::operator*()
{
  return *operator->();
}

const AllocationRecord* AllocationMap::RecordList::ConstIterator::operator->()
{
  if (!m_curr)
    UMPIRE_ERROR("Cannot dereference nullptr");
  return &m_curr->rec;
}

AllocationMap::RecordList::ConstIterator&
AllocationMap::RecordList::ConstIterator::operator++()
{
  if (!m_curr)
    UMPIRE_ERROR("Cannot dereference nullptr");
  m_curr = m_curr->prev;
  return *this;
}

AllocationMap::RecordList::ConstIterator
AllocationMap::RecordList::ConstIterator::operator++(int)
{
  ConstIterator tmp{*this};
  this->operator++();
  return tmp;
}

bool AllocationMap::RecordList::ConstIterator::operator==(
    const AllocationMap::RecordList::ConstIterator& other) const
{
  return m_list == other.m_list && m_curr == other.m_curr;
}

bool AllocationMap::RecordList::ConstIterator::operator!=(
    const AllocationMap::RecordList::ConstIterator& other) const
{
  return !(*this == other);
}

// AllocationMap
AllocationMap::AllocationMap()
    : m_block_pool{sizeof(RecordList::RecordBlock)},
      m_map{},
      m_size{0},
      m_mutex{}
{
}

void AllocationMap::insert(void* ptr, AllocationRecord record)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Inserting " << ptr);
  UMPIRE_REPLAY(
      "\"event\": \"allocation_map_insert\", \"payload\": { \"ptr\": \""
      << ptr << "\", \"record_ptr\": \"" << record.ptr
      << "\", \"record_size\": \"" << record.size
      << "\", \"record_strategy\": \"" << record.strategy << "\" }");

  auto pair = m_map.insert(ptr, *this, record);

  Map::Iterator it{pair.first};
  const bool inserted{pair.second};

  if (!inserted) {
    // Record was not added
    it->second->push_back(record);
  }
  // else
  // -> insert() already added it

  ++m_size;
}

const AllocationRecord* AllocationMap::find(void* ptr) const
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Searching for " << ptr);
  UMPIRE_REPLAY("\"event\": \"allocation_map_find\", \"payload\": { \"ptr\": \""
                << ptr << "\" }");

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
  return const_cast<AllocationRecord*>(
      const_cast<const AllocationMap*>(this)->find(ptr));
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
    const bool in_candidate = (static_cast<char*>(candidate->ptr) +
                               candidate->size) > static_cast<char*>(ptr) ||
                              (candidate->ptr == ptr);

    if (in_candidate) {
      UMPIRE_LOG(Debug, "Found " << ptr << " at " << candidate->ptr
                                 << " with size " << candidate->size);
      alloc_record = candidate;
    } else {
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
  return const_cast<AllocationRecord*>(
      const_cast<const AllocationMap*>(this)->findRecord(ptr));
}

AllocationRecord AllocationMap::remove(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  AllocationRecord ret;

  UMPIRE_LOG(Debug, "Removing " << ptr);
  UMPIRE_REPLAY(
      "\"event\": \"allocation_map_remove\", \"payload\": { \"ptr\": \""
      << ptr << "\" }");

  auto iter = m_map.find(ptr);

  if (iter->second) {
    // faster, equivalent way of checking iter != m_map->end()
    ret = iter->second->pop_back();
    if (iter->second->empty())
      m_map.removeLast();
  } else {
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
  UMPIRE_REPLAY("\"event\": \"allocation_map_clear\"");

  m_map.clear();
  m_size = 0;
}

std::size_t AllocationMap::size() const
{
  return m_size;
}

void AllocationMap::print(
    const std::function<bool(const AllocationRecord&)>&& pred,
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
        auto end_ptr = static_cast<unsigned char*>(iter->ptr) + iter->size;
        ss << iter->size << " [ " << reinterpret_cast<void*>(iter->ptr)
           << " -- " << reinterpret_cast<void*>(end_ptr) << " ] " << std::endl
#if defined(UMPIRE_ENABLE_BACKTRACE)
           << umpire::util::backtracer<trace_optional>::print(
                  iter->allocation_backtrace)
#endif // UMPIRE_ENABLE_BACKTRACE
           << std::endl;
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
  print([](const AllocationRecord&) { return true; }, os);
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

AllocationMap::ConstIterator::ConstIterator(const AllocationMap* map,
                                            iterator_begin)
    : m_outer_iter(map->m_map.begin()),
      m_inner_iter(m_outer_iter->first ? m_outer_iter->second->begin()
                                       : InnerIter{}),
      m_inner_end(m_outer_iter->first ? m_outer_iter->second->end()
                                      : InnerIter{}),
      m_outer_end(map->m_map.end())
{
}

AllocationMap::ConstIterator::ConstIterator(const AllocationMap* map,
                                            iterator_end)
    : m_outer_iter(map->m_map.end()),
      m_inner_iter(InnerIter{}),
      m_inner_end(InnerIter{}),
      m_outer_end(map->m_map.end())
{
}

const AllocationRecord& AllocationMap::ConstIterator::operator*()
{
  return m_inner_iter.operator*();
}

const AllocationRecord* AllocationMap::ConstIterator::operator->()
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

bool AllocationMap::ConstIterator::operator==(
    const AllocationMap::ConstIterator& other) const
{
  return m_outer_iter == other.m_outer_iter &&
         m_inner_iter == other.m_inner_iter;
}

bool AllocationMap::ConstIterator::operator!=(
    const AllocationMap::ConstIterator& other) const
{
  return !(*this == other);
}

} // end of namespace util
} // end of namespace umpire
