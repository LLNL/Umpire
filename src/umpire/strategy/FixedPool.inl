//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_FixedPool_INL
#define UMPIRE_FixedPool_INL

#include "umpire/strategy/FixedPool.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

template <typename T, int NP, typename IA>
void 
FixedPool<T, NP, IA>::newPool(struct Pool **pnew) {
  struct Pool *p = static_cast<struct Pool *>(IA::allocate(sizeof(struct Pool) + NP * sizeof(unsigned int)));
  p->numAvail = m_num_per_pool;
  p->next = NULL;

  p->data  = reinterpret_cast<unsigned char*>(m_allocator->allocate(m_num_per_pool * sizeof(T)));
  p->avail = reinterpret_cast<unsigned int *>(p + 1);
  for (int i = 0; i < NP; i++) p->avail[i] = -1;

  *pnew = p;

  m_current_size += m_num_per_pool*sizeof(T);
  if (m_current_size > m_highwatermark) {
    m_highwatermark = m_current_size;
  }
}

template <typename T, int NP, typename IA>
T* 
FixedPool<T, NP, IA>::allocInPool(struct Pool *p) {
    if (!p->numAvail) return NULL;

    for (int i = 0; i < NP; i++) {
      const int bit = ffs(p->avail[i]) - 1;
      if (bit >= 0) {
        p->avail[i] ^= 1 << bit;
        p->numAvail--;
        const int entry = i * sizeof(unsigned int) * 8 + bit;
        return reinterpret_cast<T*>(p->data) + entry;
      }
    }

    return NULL;
}

template <typename T, int NP, typename IA>
FixedPool<T, NP, IA>::FixedPool(
    const std::string& name,
    int id,
    Allocator allocator) : 
  AllocationStrategy(name, id),
  m_num_per_pool(NP * sizeof(unsigned int) * 8),
  m_total_pool_size(sizeof(struct Pool) + m_num_per_pool * sizeof(T) + NP * sizeof(unsigned int)),
  m_num_blocks(0),
  m_highwatermark(0),
  m_current_size(0),
  m_allocator(allocator.getAllocationStrategy())
{ 
  newPool(&m_pool); 
}

template <typename T, int NP, typename IA>
FixedPool<T, NP, IA>::~FixedPool() {
    for (struct Pool *curr = m_pool; curr; ) {
      struct Pool *next = curr->next;
      m_allocator->deallocate(curr);
      m_current_size -= sizeof(T)*m_num_per_pool;
      curr = next;
    }
  }

template <typename T, int NP, typename IA>
void* 
FixedPool<T, NP, IA>::allocate(size_t bytes) {
  T* ptr = NULL;

  struct Pool *prev = NULL;
  struct Pool *curr = m_pool;
  while (!ptr && curr) {
    ptr = allocInPool(curr);
    prev = curr;
    curr = curr->next;
  }

  if (!ptr) {
    newPool(&prev->next);
    ptr = static_cast<T*>(allocate(bytes));
    // TODO: In this case we should reverse the linked list for optimality
  }
  else {
    m_num_blocks++;
  }

  ResourceManager::getInstance().registerAllocation(ptr, new util::AllocationRecord{ptr, sizeof(T), this->shared_from_this()});

  return ptr;
}

template <typename T, int NP, typename IA>
void 
FixedPool<T,NP, IA>::deallocate(void* ptr) {
  T* t_ptr = static_cast<T*>(ptr);

  int i = 0;
  for (struct Pool *curr = m_pool; curr; curr = curr->next) {
    const T* start = reinterpret_cast<T*>(curr->data);
    const T* end   = reinterpret_cast<T*>(curr->data) + m_num_per_pool;
    if ( (t_ptr >= start) && (t_ptr < end) ) {
      // indexes bits 0 - m_num_per_pool-1
      const int indexD = t_ptr - reinterpret_cast<T*>(curr->data);
      const int indexI = indexD / ( sizeof(unsigned int) * 8 );
      const int indexB = indexD % ( sizeof(unsigned int) * 8 );
#ifndef NDEBUG
      if ((curr->avail[indexI] & (1 << indexB))) {
        std::cerr << "Trying to deallocate an entry that was not marked as allocated" << std::endl;
      }
#endif
      curr->avail[indexI] ^= 1 << indexB;
      curr->numAvail++;
      m_num_blocks--;
      ResourceManager::getInstance().deregisterAllocation(ptr);

      return;
    }
    i++;
  }

  UMPIRE_ERROR("Could not find pointer to deallocate");
}

template <typename T, int NP, typename IA>
long 
FixedPool<T, NP, IA>::getCurrentSize() {
    return m_current_size;
}

template <typename T, int NP, typename IA>
long 
FixedPool<T, NP, IA>::getHighWatermark() {
  return m_highwatermark;
}

template <typename T, int NP, typename IA>
size_t
FixedPool<T, NP, IA>::numPools() const {
  std::size_t np = 0;
  for (struct Pool *curr = m_pool; curr; curr = curr->next) np++;
  return np;
}

template <typename T, int NP, typename IA>
Platform 
FixedPool<T, NP, IA>::getPlatform()
{ 
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_FixedPool_INL
