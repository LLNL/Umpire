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

// Judy: number of Integers in a key
static const unsigned int judy_depth = 1;

// Judy: max height of stack
static const unsigned int judy_max_levels = sizeof(uintptr_t);

// Judy: length of key in bytes
static const unsigned int judy_max = judy_depth * JUDY_key_size;

namespace umpire {
namespace util {

template <typename T>
struct ListBlock
{
  T rec;
  ListBlock* prev;
};

static umpire::util::FixedMallocPool block_pool(sizeof(ListBlock<AllocationRecord>));

class RecordListConstIterator;

class RecordList
{
public:
  using Block = ListBlock<AllocationRecord>;

  // Iterator needs access to m_tail
  friend RecordListConstIterator;

  RecordList(const AllocationRecord& record);
  ~RecordList();

  void push_back(AllocationRecord rec);
  AllocationRecord pop_back();

  RecordListConstIterator begin() const;
  RecordListConstIterator end() const;

  size_t size() const;
  bool empty() const;
  AllocationRecord* back();

private:
  Block* m_tail;
  size_t m_length;
};

static umpire::util::FixedMallocPool list_pool(sizeof(RecordList));

// Iterator for RecordList
class RecordListConstIterator : public std::iterator<std::forward_iterator_tag, AllocationRecord>
{
  const RecordList *m_list;
  RecordList::Block* m_curr;

public:
  RecordListConstIterator(const RecordList* list, bool end);
  RecordListConstIterator(const RecordListConstIterator& other) = default;

  const AllocationRecord& operator*() const;
  const AllocationRecord* operator->() const;
  RecordListConstIterator& operator++();
  RecordListConstIterator operator++(int);

  bool operator==(const RecordListConstIterator& other);
  bool operator!=(const RecordListConstIterator& other);
};

// Record List
RecordList::RecordList(const AllocationRecord& record)
  : m_tail(nullptr), m_length(0)
{
  push_back(record);
  m_tail->prev = nullptr;
}

RecordList::~RecordList()
{
  Block* curr = m_tail;
  while (curr) {
    Block* prev = curr->prev;
    block_pool.deallocate(curr);
    curr = prev;
  }
}

void RecordList::push_back(AllocationRecord rec)
{
  Block* curr = reinterpret_cast<Block*>(block_pool.allocate());
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
    Block* prev = m_tail->prev;
    block_pool.deallocate(m_tail);
    m_tail = prev;
    m_length--;
  }
  return ret;
}

RecordListConstIterator RecordList::begin() const
{
  return RecordListConstIterator{this, false};
}

RecordListConstIterator RecordList::end() const
{
  return RecordListConstIterator{this, true};
}

size_t RecordList::size() const { return m_length; }
bool RecordList::empty() const { return size() == 0; }

AllocationRecord* RecordList::back() { return &m_tail->rec; }

// RecordListConstIterator
RecordListConstIterator::RecordListConstIterator(const RecordList* list,
                                                 bool end)
  : m_list(list), m_curr(end ? nullptr : m_list->m_tail)
{
}

const AllocationRecord& RecordListConstIterator::operator*() const
{
  if (!m_curr) UMPIRE_ERROR("Cannot dereference nullptr");
  return m_curr->rec;
}

const AllocationRecord* RecordListConstIterator::operator->() const
{
  return m_curr ? &(m_curr->rec) : nullptr;
}

RecordListConstIterator& RecordListConstIterator::operator++()
{
  m_curr = m_curr->prev;
  return *this;
}

RecordListConstIterator RecordListConstIterator::operator++(int)
{
  RecordListConstIterator tmp{*this};
  m_curr = m_curr->prev;
  return tmp;
}

bool RecordListConstIterator::operator==(const RecordListConstIterator& other)
{
  return m_list == other.m_list && m_curr == other.m_curr;
}

bool RecordListConstIterator::operator!=(const RecordListConstIterator& other)
{
  return !(*this == other);
}

// AllocationMap
AllocationMap::AllocationMap() :
  m_array(nullptr),
  m_size(0),
  m_last(nullptr),
  m_mutex()
{
  // Create new judy array
  m_array = judy_open(judy_max_levels, judy_depth);
}

AllocationMap::~AllocationMap()
{
  // Delete all entries
  clear();

  // Close the judy array, freeing all memory.
  judy_close(m_array);
}

void AllocationMap::insert(void* ptr, AllocationRecord record)
{
  UMPIRE_LOCK;
  UMPIRE_LOG(Debug, "Inserting " << ptr);

  // Find the key
  m_last = judy_cell(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  UMPIRE_ASSERT(m_last);

  auto plist = reinterpret_cast<RecordList**>(m_last);

  if (!*plist) {
    // if there is no list there, create one and emplace the record
    (*plist) = new (list_pool.allocate()) RecordList{record};
  }
  else {
    // else, push onto that list
    (*plist)->push_back(record);
  }
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
  AllocationRecord* alloc_record{nullptr};

  UMPIRE_LOCK;

  // Seek and find key (key = parent_ptr)
  uintptr_t parent_ptr{0};
  m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);
  judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);

  // The record list at that ptr
  auto list = m_last ? reinterpret_cast<RecordList*>(*m_last) : nullptr;

  // If the ptrs do not match, or the key does not exist, get the previous entry
  if (parent_ptr != reinterpret_cast<uintptr_t>(ptr) || !list)
  {
    m_last = judy_prv(m_array);
    // Find key associated to this one
    judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), judy_max);
  }

  // Update record list
  list = m_last ? reinterpret_cast<RecordList*>(*m_last) : nullptr;

  // If a list was found, return its tail
  if (list) {
    alloc_record = list->back();

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

    // Locate ptr
    m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), judy_max);

    // If found, remove it
    if (m_last && *m_last) {
      RecordList* list = reinterpret_cast<RecordList*>(*m_last);
      UMPIRE_ASSERT(list->size());

      ret = list->pop_back();
      --m_size;

      if (list->empty()) {
        // Manually call destructor
        list->~RecordList();

        // Mark as deallocated in the pool
        list_pool.deallocate(list);

        // Remove entry from judy array
        m_last = judy_del(m_array);
      }
    } else {
      UMPIRE_ERROR("Cannot remove " << ptr );
    }

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
  uintptr_t key = 0;

  // Loop over the level 0 tree
  while((m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0)) != nullptr) {
    auto list = reinterpret_cast<RecordList*>(*m_last);

    // Manually call destructor
    list->~RecordList();

    // Mark as deallocated in the pool
    list_pool.deallocate(list);

    // Delete the key and cell for the current stack entry.
    judy_del(m_array);
  }
}

size_t AllocationMap::size() const { return m_size; }

void
AllocationMap::print(const std::function<bool (const AllocationRecord&)>&& pred,
                     std::ostream& os) const
{
  uintptr_t key = 0;
  for(m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0);
      m_last != nullptr;
      m_last = judy_nxt(m_array)) {
    auto list = reinterpret_cast<RecordList*>(*m_last);

    void* addr;
    judy_key(m_array, reinterpret_cast<unsigned char*>(&addr), judy_max);

    std::stringstream ss;
    bool any_match = false;
    ss << addr << " {" << std::endl;
    for (auto iter{list->begin()}; iter != list->end(); ++iter) {
      if (pred(*iter)) {
        any_match = true;
        ss << iter->size <<
          " [ " << reinterpret_cast<void*>(iter->ptr) <<
          " -- " << reinterpret_cast<void*>(static_cast<unsigned char*>(iter->ptr)+iter->size) <<
          " ] " << std::endl;
      }
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


AllocationMapConstIterator AllocationMap::begin() const
{
  return AllocationMapConstIterator{this, false};
}

AllocationMapConstIterator AllocationMap::end() const
{
  return AllocationMapConstIterator{this, true};
}

// AllocationMapConstIterator
AllocationMapConstIterator::AllocationMapConstIterator(const AllocationMap* map,
                                                       bool end)
  : m_array(map->m_array), m_last(nullptr), m_ptr(0), m_iter(nullptr)
{
  if (!end) {
    m_last = judy_strt(m_array, reinterpret_cast<const unsigned char*>(&m_ptr), judy_max);
  }
  else {
    m_last = judy_end(m_array);
  }

  if (m_last) {
    auto list = reinterpret_cast<RecordList*>(*m_last);
    m_iter = new RecordListConstIterator{list, end};
  }
}

AllocationMapConstIterator::AllocationMapConstIterator(const AllocationMap* map,
                                                       uintptr_t ptr)
  : m_array(map->m_array), m_last(nullptr), m_ptr(ptr), m_iter(nullptr)
{
  m_last = judy_strt(m_array, reinterpret_cast<const unsigned char*>(&m_ptr), judy_max);
  if (m_last) {
    auto list = reinterpret_cast<RecordList*>(*m_last);
    m_iter = new RecordListConstIterator{list, false};
  }
}

AllocationMapConstIterator::~AllocationMapConstIterator()
{
  delete m_iter;
}

const AllocationRecord& AllocationMapConstIterator::operator*() const
{
  return m_iter->operator*();
}

const AllocationRecord* AllocationMapConstIterator::operator->() const
{
  return m_iter->operator->();
}

AllocationMapConstIterator& AllocationMapConstIterator::operator++()
{
  auto list = reinterpret_cast<RecordList*>(*m_last);
  (*m_iter)++;
  if (*m_iter == list->end()) {
    // Move to a new pointer
    JudySlot* new_slot = judy_strt(m_array, reinterpret_cast<const unsigned char*>(&m_ptr), judy_max);
    if (new_slot && (m_last != new_slot)) {
      m_last = new_slot;
      list = reinterpret_cast<RecordList*>(*m_last);
      *m_iter = RecordListConstIterator{list, false};
    }
  }

  return *this;
}

AllocationMapConstIterator AllocationMapConstIterator::operator++(int)
{
  AllocationMapConstIterator tmp{*this};
  ++(*this);
  return tmp;
}

bool AllocationMapConstIterator::operator==(const AllocationMapConstIterator& other)
{
  return (m_array == other.m_array && m_ptr == other.m_ptr && *m_iter == *other.m_iter);
}

bool AllocationMapConstIterator::operator!=(const AllocationMapConstIterator& other)
{
  return !(*this == other);
}

} // end of namespace util
} // end of namespace umpire
