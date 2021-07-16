//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/detail/allocation_map.hpp"
#include "umpire/detail/fixed_malloc_pool.hpp"
#include "umpire/detail/log.hpp"
#include "umpire/detail/replay.hpp"

#if defined(UMPIRE_ENABLE_BACKTRACE)
#include "umpire/util/backtrace.hpp"
#endif

#include <sstream>
#include <type_traits>
#include <utility>

namespace umpire {
namespace detail {

// Record List
allocation_map::record_list::record_list(allocation_map& map, allocation_record record) :
  m_map{map},
  m_tail{nullptr},
  m_length{0}
{
  push_back(record);
}

allocation_map::record_list::~record_list()
{
  record_block* curr = m_tail;
  while (curr) {
    record_block* prev = curr->prev;
    m_map.m_block_pool.deallocate(curr);
    curr = prev;
  }
}

void
allocation_map::record_list::push_back(const allocation_record& rec)
{
  record_block* curr = static_cast<record_block*>(m_map.m_block_pool.allocate());
  curr->prev = m_tail;
  curr->rec = rec;
  m_tail = curr;
  m_length++;
}

allocation_record
allocation_map::record_list::pop_back()
{
  if (m_length == 0) {
    UMPIRE_ERROR("pop_back() called, but m_length == 0");
  }

  const allocation_record ret = m_tail->rec;
  record_block* prev = m_tail->prev;

  // Deallocate and move tail pointer
  m_map.m_block_pool.deallocate(m_tail);
  m_tail = prev;

  // Reduce size
  m_length--;

  return ret;
}

allocation_map::record_list::const_iterator
allocation_map::record_list::begin() const
{
  return allocation_map::record_list::const_iterator{this, iterator_begin{}};
}

allocation_map::record_list::const_iterator
allocation_map::record_list::end() const
{
  return allocation_map::record_list::const_iterator{this, iterator_end{}};
}

std::size_t
allocation_map::record_list::size() const
{
  return m_length;
}

bool
allocation_map::record_list::empty() const
{
  return size() == 0;
}

allocation_record*
allocation_map::record_list::back()
{
  return &m_tail->rec;
}

const allocation_record*
allocation_map::record_list::back() const
{
  return &m_tail->rec;
}

allocation_map::record_list::const_iterator::const_iterator()
  : m_list(nullptr), m_curr(nullptr)
{
}

allocation_map::record_list::const_iterator::const_iterator(
  const record_list* list, iterator_begin)
  : m_list(list), m_curr(m_list->m_tail)
{
}

allocation_map::record_list::const_iterator::const_iterator(
  const record_list* list, iterator_end)
  : m_list(list), m_curr(nullptr)
{
}

const allocation_record&
allocation_map::record_list::const_iterator::operator*()
{
  return *operator->();
}

const allocation_record*
allocation_map::record_list::const_iterator::operator->()
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  return &m_curr->rec;
}

allocation_map::record_list::const_iterator&
allocation_map::record_list::const_iterator::operator++()
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  m_curr = m_curr->prev;
  return *this;
}

allocation_map::record_list::const_iterator
allocation_map::record_list::const_iterator::operator++(int)
{
  const_iterator tmp{*this};
  this->operator++();
  return tmp;
}

bool
allocation_map::record_list::const_iterator::operator==(
  const allocation_map::record_list::const_iterator& other) const
{
  return m_list == other.m_list && m_curr == other.m_curr;
}

bool
allocation_map::record_list::const_iterator::operator!=(
  const allocation_map::record_list::const_iterator& other) const
{
  return !(*this == other);
}

// allocation_map
allocation_map::allocation_map() :
  m_block_pool{sizeof(record_list::record_block)},
  m_map{},
  m_size{0},
  m_mutex{}
{
}

void
allocation_map::insert(void* ptr, allocation_record record)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Inserting " << ptr);
  UMPIRE_REPLAY("\"event\": \"allocation_map_insert\", \"payload\": { \"ptr\": \"" << ptr << "\", \"record_ptr\": \"" << record.ptr << "\", \"record_size\": \"" << record.size << "\", \"record_strategy\": \"" << record.strategy << "\" }");

  auto pair = m_map.insert(ptr, *this, record);

  map::Iterator it{pair.first};
  const bool inserted{pair.second};

  if (!inserted) {
    // Record was not added
    it->second->push_back(record);
  }
  // else
  // -> insert() already added it

  ++m_size;
}

const allocation_record*
allocation_map::find(void* ptr) const
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Searching for " << ptr);
  UMPIRE_REPLAY("\"event\": \"allocation_map_find\", \"payload\": { \"ptr\": \"" << ptr << "\" }");

  const allocation_record* alloc_record = doFindRecord(ptr);

  if (alloc_record) {
    return alloc_record;
  } else {
#if !defined(NDEBUG)
    // use this from a debugger to dump the contents of the allocation_map
    printAll();
#endif
    UMPIRE_ERROR("Allocation not mapped: " << ptr);
  }
}

allocation_record*
allocation_map::find(void* ptr)
{
  return const_cast<allocation_record*>(const_cast<const allocation_map*>(this)->find(ptr));
}

const allocation_record*
allocation_map::doFindRecord(void* ptr) const noexcept
{
  const allocation_record* alloc_record = nullptr;

  map::ConstIterator iter = m_map.findOrBefore(ptr);

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

const allocation_record*
allocation_map::findRecord(void* ptr) const noexcept
{
  std::lock_guard<std::mutex> lock(m_mutex);

  // Call method
  return doFindRecord(ptr);
}

allocation_record*
allocation_map::findRecord(void* ptr) noexcept
{
  return const_cast<allocation_record*>(const_cast<const allocation_map*>(this)->findRecord(ptr));
}


allocation_record
allocation_map::remove(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  allocation_record ret;

  UMPIRE_LOG(Debug, "Removing " << ptr);
  UMPIRE_REPLAY("\"event\": \"allocation_map_remove\", \"payload\": { \"ptr\": \"" << ptr << "\" }");

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

bool
allocation_map::contains(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
  return (findRecord(ptr) != nullptr);
}

void
allocation_map::clear()
{
  std::lock_guard<std::mutex> lock(m_mutex);

  UMPIRE_LOG(Debug, "Clearing");
  UMPIRE_REPLAY("\"event\": \"allocation_map_clear\"");

  m_map.clear();
  m_size = 0;
}

std::size_t
allocation_map::size() const
{
  return m_size;
}

void
allocation_map::print(const std::function<bool (const allocation_record&)>&& pred,
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
        auto end_ptr = static_cast<unsigned char*>(iter->ptr)+iter->size;
        ss << iter->size <<
          " [ " << reinterpret_cast<void*>(iter->ptr) <<
          " -- " << reinterpret_cast<void*>(end_ptr) <<
          " ] " << std::endl
#if defined(UMPIRE_ENABLE_BACKTRACE)
          << umpire::util::backtracer<trace_optional>::print(iter->allocation_backtrace)
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

void
allocation_map::printAll(std::ostream& os) const
{
  os << "🔍 Printing allocation map contents..." << std::endl;
  print([] (const allocation_record&) { return true; }, os);
  os << "done." << std::endl;
}


allocation_map::const_iterator
allocation_map::begin() const
{
  return allocation_map::const_iterator{this, iterator_begin{}};
}

allocation_map::const_iterator
allocation_map::end() const
{
  return allocation_map::const_iterator{this, iterator_end{}};
}

allocation_map::const_iterator::const_iterator(
  const allocation_map* map, iterator_begin) :
  m_outer_iter(map->m_map.begin()),
  m_inner_iter(m_outer_iter->first ? m_outer_iter->second->begin() : inner_iter{}),
  m_inner_end(m_outer_iter->first ? m_outer_iter->second->end() : inner_iter{}),
  m_outer_end(map->m_map.end())
{
}

allocation_map::const_iterator::const_iterator(
  const allocation_map* map, iterator_end) :
  m_outer_iter(map->m_map.end()),
  m_inner_iter(inner_iter{}),
  m_inner_end(inner_iter{}),
  m_outer_end(map->m_map.end())
{
}

const allocation_record&
allocation_map::const_iterator::operator*()
{
  return m_inner_iter.operator*();
}

const allocation_record*
allocation_map::const_iterator::operator->()
{
  return m_inner_iter.operator->();
}

allocation_map::const_iterator&
allocation_map::const_iterator::operator++()
{
  ++m_inner_iter;
  if (m_inner_iter == m_inner_end) {
    ++m_outer_iter;
    if (m_outer_iter != m_outer_end) {
      m_inner_iter = m_outer_iter->second->begin();
      m_inner_end = m_outer_iter->second->end();
    } else {
      m_inner_iter = inner_iter{};
    }
  }
  return *this;
}

allocation_map::const_iterator
allocation_map::const_iterator::operator++(int)
{
  const_iterator tmp{*this};
  ++(*this);
  return tmp;
}

bool
allocation_map::const_iterator::operator==(
  const allocation_map::const_iterator& other) const
{
  return
    m_outer_iter == other.m_outer_iter &&
    m_inner_iter == other.m_inner_iter;
}

bool
allocation_map::const_iterator::operator!=(
  const allocation_map::const_iterator& other) const
{
  return !(*this == other);
}

} // end of namespace util
} // end of namespace umpire
