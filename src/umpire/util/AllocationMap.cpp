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

#include <iterator>
#include <sstream>

namespace umpire {
namespace util {

namespace {

template <typename T>
struct ListBlock
{
  T rec;
  ListBlock* prev;
};

static umpire::util::FixedMallocPool block_pool(sizeof(ListBlock<AllocationRecord>));

// Forward declare const iterator
class RecordListConstIterator;

// TODO Profile this to determine if we should add a small string
// optimization on top of the block pooling.
class RecordList
{
public:
  using Block = ListBlock<AllocationRecord>;
  friend RecordListConstIterator;

  RecordList() : m_tail(nullptr), m_length(0) {}

  RecordList(const AllocationRecord& record) : m_tail(nullptr), m_length(0) {
    push_back(record);
    m_tail->prev = nullptr;
  }

  ~RecordList() {
    Block* curr = m_tail;
    while (curr) {
      Block* prev = curr->prev;
      block_pool.deallocate(curr);
      curr = prev;
    }
  }

  void push_back(AllocationRecord rec) {
    Block* curr = reinterpret_cast<Block*>(block_pool.allocate());
    curr->prev = m_tail;
    curr->rec = rec;
    m_tail = curr;
    m_length++;
  }

  AllocationRecord pop_back() {
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

  RecordListConstIterator begin() const;
  RecordListConstIterator end() const;

  size_t size() const { return m_length; }
  bool empty() const { return size() == 0; }

  AllocationRecord* back() { return &m_tail->rec; }

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
  RecordListConstIterator(const RecordList* list, RecordList::Block* curr, bool end) :
    m_list(list), m_curr(curr) {
    if (end) {
      // Skip to the head of the list
      while(m_curr->prev != nullptr) m_curr = m_curr->prev;
    }
  }

  RecordListConstIterator(const RecordListConstIterator& other) = default;


  AllocationRecord& operator*() const {
    return m_curr->rec;
  }

  const AllocationRecord* operator->() const {
    return &(m_curr->rec);
  }

  RecordListConstIterator& operator++() {
    m_curr = m_curr->prev;
    return *this;
  }

  RecordListConstIterator operator++(int) {
    RecordListConstIterator tmp{*this};
    m_curr = m_curr->prev;
    return tmp;
  }

  bool operator==(const RecordListConstIterator& other) {
    return m_list == other.m_list && m_curr == other.m_curr;
  }

  bool operator!=(const RecordListConstIterator& other) {
    return !(*this == other);
  }
};

RecordListConstIterator RecordList::begin() const {
  return RecordListConstIterator{this, m_tail, false};
}

RecordListConstIterator RecordList::end() const {
  return RecordListConstIterator{this, m_tail, true};
}


} // end of anonymous namespace

static inline const unsigned char* to_judy_buff(void*& key) {
  return reinterpret_cast<const unsigned char*>(&key);
}

AllocationMap::AllocationMap() :
  m_array(nullptr),
  m_last(nullptr),
  m_max_levels(sizeof(uintptr_t)),
  m_depth(1),
  m_mutex(new std::mutex())
{
  // Create new judy array
  m_array = judy_open(m_max_levels, m_depth);
}

AllocationMap::~AllocationMap()
{
  // Delete all entries
  clear();

  // Close the judy array, freeing all memory.
  judy_close(m_array);

  // Delete the mutex
  delete m_mutex;
}

void AllocationMap::insert(void* ptr, AllocationRecord record)
{
  try {
    UMPIRE_LOCK;
    UMPIRE_LOG(Debug, "Inserting " << ptr);

    // Find the key
    m_last = judy_cell(m_array, to_judy_buff(ptr), m_depth * JUDY_key_size);
    UMPIRE_ASSERT(m_last);

    auto plist = reinterpret_cast<RecordList**>(m_last);
    if (!*plist) (*plist) = new (list_pool.allocate()) RecordList{record};
    else (*plist)->push_back(record);

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
}

AllocationRecord* AllocationMap::find(void* ptr) const
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


AllocationRecord* AllocationMap::findRecord(void* ptr) const
{
  AllocationRecord* alloc_record = nullptr;

  try {
    UMPIRE_LOCK;
    uintptr_t parent_ptr;

    // Seek and find key (key = parent_ptr)
    m_last = judy_strt(m_array, to_judy_buff(ptr), m_depth * JUDY_key_size);
    judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), m_depth * JUDY_key_size);

    // If not found, get the previous one
    // TODO Start fixing here
    if (parent_ptr != reinterpret_cast<uintptr_t>(ptr) || reinterpret_cast<RecordList*>(*m_last) == nullptr)
    {
      m_last = judy_prv(m_array);
      // Find key associated to this one
      judy_key(m_array, reinterpret_cast<unsigned char*>(&parent_ptr), m_depth * JUDY_key_size);
    }

    auto list = m_last ? reinterpret_cast<RecordList*>(*m_last) : nullptr;

    // If a value was found
    if (list) {
      alloc_record = list->back();

      if (alloc_record && ((parent_ptr + alloc_record->m_size) > reinterpret_cast<uintptr_t>(ptr))) {
         UMPIRE_LOG(Debug, "Found " << ptr << " at " << parent_ptr
                    << " with size " << alloc_record->m_size);
      }
      else {
         alloc_record = nullptr;
      }
    }
    UMPIRE_UNLOCK;
  }
  catch (...){
    UMPIRE_UNLOCK;
    throw;
  }

  return alloc_record;
}

AllocationRecord AllocationMap::remove(void* ptr)
{
  AllocationRecord ret{};

  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "Removing " << ptr);

    // Locate ptr
    m_last = judy_slot(m_array, to_judy_buff(ptr), m_depth * JUDY_key_size);

    // If found, remove it
    if (m_last && *m_last) {
      RecordList* list = reinterpret_cast<RecordList*>(*m_last);
      UMPIRE_ASSERT(list->size());

      ret = list->pop_back();

      if (list->empty()) {
        if((m_last = judy_slot(m_array, reinterpret_cast<unsigned char*>(&ptr), m_depth * JUDY_key_size)) != nullptr) {
          auto list = reinterpret_cast<RecordList*>(*m_last);

          // Manually call destructor
          list->~RecordList();

          // Mark as deallocated in the pool
          list_pool.deallocate(list);

          m_last = judy_del(m_array);
        }
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
  // TODO Why is max = 0 here?
  while((m_last = judy_strt(m_array, reinterpret_cast<unsigned char*>(&key), 0)) != nullptr) {
    auto list = reinterpret_cast<RecordList*>(*m_last);

    // Manually call destructor
    list->~RecordList();

    // Mark as deallocated in the pool
    list_pool.deallocate(list);

    // delete the key and cell for the current stack entry.
    judy_del(m_array);
  }
}

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
    judy_key(m_array, reinterpret_cast<unsigned char*>(&addr), m_depth * JUDY_key_size);

    std::stringstream ss;
    bool any_match = false;
    ss << addr << " {" << std::endl;
    for (auto iter = list->begin(); iter != list->end(); iter++) {
      if (pred(*iter)) {
        any_match = true;
        ss << iter->m_size <<
          " [ " << reinterpret_cast<void*>(iter->m_ptr) <<
          " -- " << reinterpret_cast<void*>(static_cast<unsigned char*>(iter->m_ptr)+iter->m_size) <<
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

} // end of namespace util
} // end of namespace umpire
