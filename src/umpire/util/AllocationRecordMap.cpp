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
#include "umpire/util/AllocationRecordMap.hpp"

#include "umpire/util/FixedMallocPool.hpp"

#include "umpire/util/Macros.hpp"

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

// TODO Profile this to determine if we should add a small string
// optimization on top of the block pooling.
class RecordList
{
public:
  using RecordListBlock = ListBlock<AllocationRecord>;

  RecordList() : m_tail(nullptr) {}

  ~RecordList() {
    RecordListBlock* curr = m_tail;
    while (curr) {
      RecordListBlock* prev = curr->prev;
      block_pool.deallocate(curr);
      curr = prev;
    }
  }

  void push_back(AllocationRecord rec) {
    RecordListBlock* curr = reinterpret_cast<RecordListBlock*>(block_pool.allocate());
    curr->prev = m_tail;
    curr->rec = rec;
    m_tail = curr;
    m_length++;
  }

  AllocationRecord pop_back() {
    AllocationRecord ret = m_tail->rec;
    RecordListBlock* prev = m_tail->prev;
    block_pool.deallocate(m_tail);
    m_tail = prev;
    m_length--;
    return ret;
  }

  size_t size() const { return m_length; }
  bool empty() const { return size() == 0; }

  AllocationRecord* back() { return &m_tail->rec; }

private:
  RecordListBlock* m_tail;
  size_t m_length;
};

static umpire::util::FixedMallocPool list_pool(sizeof(RecordList));

} // end of anonymous namespace

static inline const unsigned char* to_judy_buff(void*& key) {
  return reinterpret_cast<const unsigned char*>(&key);
}

AllocationRecordMap::AllocationRecordMap() :
  m_array(nullptr),
  m_last(nullptr),
  m_max_levels(sizeof(uintptr_t)),
  m_depth(1),
  m_mutex(new std::mutex())
{
  // Create new judy array
  m_array = judy_open(m_max_levels, m_depth);
}

AllocationRecordMap::~AllocationRecordMap()
{
  // Delete all entries
  clear();

  // Close the judy array, freeing all memory.
  judy_close(m_array);

  delete m_mutex;
}

void AllocationRecordMap::insert(void* ptr, AllocationRecord record)
{
  try {
    UMPIRE_LOCK;
    UMPIRE_LOG(Debug, "Inserting " << ptr);

    // Find the key
    m_last = judy_cell(m_array, to_judy_buff(ptr), m_depth * JUDY_key_size);
    UMPIRE_ASSERT(m_last);

    auto plist = reinterpret_cast<RecordList**>(m_last);
    if (!*plist) (*plist) = new (list_pool.allocate()) RecordList;
    (*plist)->push_back(record);

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
}

AllocationRecord* AllocationRecordMap::find(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
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

AllocationRecord AllocationRecordMap::remove(void* ptr)
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

bool AllocationRecordMap::contains(void* ptr) const
{
  UMPIRE_LOG(Debug, "Searching for " << ptr);
  return (find(ptr) != nullptr);
}

void AllocationRecordMap::clear()
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
AllocationRecordMap::print(const std::function<bool (const AllocationRecord*)>&& UMPIRE_UNUSED_ARG(pred),
                           std::ostream& UMPIRE_UNUSED_ARG(os)) const
{
  UMPIRE_ERROR("TBD");
}

void AllocationRecordMap::printAll(std::ostream& UMPIRE_UNUSED_ARG(os)) const
{
  UMPIRE_ERROR("TBD");
}

} // end of namespace util
} // end of namespace umpire
