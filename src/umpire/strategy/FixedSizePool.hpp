//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef _FIXEDSIZEPOOL_HPP
#define _FIXEDSIZEPOOL_HPP

#include <stdio.h>

#include <cstring>
#include <iostream>

#include "StdAllocator.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4245)
#pragma warning(disable : 4267)
#endif

inline int find_first_set(int i)
{
#if defined(_MSC_VER)
  unsigned long bit;
  unsigned long i_l = static_cast<unsigned long>(i);
  _BitScanForward(&bit, i_l);
  return static_cast<int>(bit);
#else
  return ffs(i);
#endif
}

template <class T, class MA, class IA = StdAllocator, int NP = (1 << 6)>
class FixedSizePool {
 protected:
  struct Pool {
    unsigned char *data;
    unsigned int *avail;
    unsigned int numAvail;
    struct Pool *next;
  };

  struct Pool *pool;
  const std::size_t numPerPool;
  const std::size_t totalPoolSize;

  std::size_t numBlocks;

  void newPool(struct Pool **pnew)
  {
    struct Pool *p = static_cast<struct Pool *>(
        IA::allocate(sizeof(struct Pool) + NP * sizeof(unsigned int)));
    p->numAvail = numPerPool;
    p->next = NULL;

    p->data =
        reinterpret_cast<unsigned char *>(MA::allocate(numPerPool * sizeof(T)));
    p->avail = reinterpret_cast<unsigned int *>(p + 1);
    for (int i = 0; i < NP; i++)
      p->avail[i] = (~0);

    *pnew = p;
  }

  T *allocInPool(struct Pool *p)
  {
    if (!p->numAvail)
      return NULL;

    for (int i = 0; i < NP; i++) {
      const int bit = find_first_set(p->avail[i]) - 1;
      if (bit >= 0) {
        p->avail[i] ^= 1 << bit;
        p->numAvail--;
        const int entry = i * sizeof(unsigned int) * 8 + bit;
        return reinterpret_cast<T *>(p->data) + entry;
      }
    }

    return NULL;
  }

 public:
  static inline FixedSizePool &getInstance()
  {
    static FixedSizePool instance;
    return instance;
  }

  FixedSizePool()
      : numPerPool(NP * sizeof(unsigned int) * 8),
        totalPoolSize(sizeof(struct Pool) + numPerPool * sizeof(T) +
                      NP * sizeof(unsigned int)),
        numBlocks(0)
  {
    newPool(&pool);
  }

  ~FixedSizePool()
  {
    for (struct Pool *curr = pool; curr;) {
      struct Pool *next = curr->next;
      MA::deallocate(curr->data);
      IA::deallocate(curr);
      curr = next;
    }
  }

  T *allocate()
  {
    T *ptr = NULL;

    struct Pool *prev = NULL;
    struct Pool *curr = pool;
    while (!ptr && curr) {
      ptr = allocInPool(curr);
      prev = curr;
      curr = curr->next;
    }

    if (!ptr) {
      newPool(&prev->next);
      ptr = allocate();
      // TODO: In this case we should reverse the linked list for optimality
    } else {
      numBlocks++;
    }
    return ptr;
  }

  void deallocate(T *ptr)
  {
    int i = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next) {
      const T *start = reinterpret_cast<T *>(curr->data);
      const T *end = reinterpret_cast<T *>(curr->data) + numPerPool;
      if ((ptr >= start) && (ptr < end)) {
        // indexes bits 0 - numPerPool-1
        const int indexD = ptr - reinterpret_cast<T *>(curr->data);
        const int indexI = indexD / (sizeof(unsigned int) * 8);
        const int indexB = indexD % (sizeof(unsigned int) * 8);
#ifndef NDEBUG
        if ((curr->avail[indexI] & (1 << indexB))) {
          std::cerr << "Trying to deallocate an entry that was not marked as "
                       "allocated"
                    << std::endl;
        }
#endif
        curr->avail[indexI] ^= 1 << indexB;
        curr->numAvail++;
        numBlocks--;
        return;
      }
      i++;
    }
    std::cerr << "Could not find pointer to deallocate" << std::endl;
    throw(std::bad_alloc());
  }

  /// Return allocated size to user.
  std::size_t getCurrentSize() const
  {
    return numBlocks * sizeof(T);
  }

  /// Return total size with internal overhead.
  std::size_t getActualSize() const
  {
    return numPools() * totalPoolSize;
  }

  /// Return the number of pools
  std::size_t numPools() const
  {
    std::size_t np = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next)
      np++;
    return np;
  }

  /// Return the pool size
  std::size_t poolSize() const
  {
    return totalPoolSize;
  }
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif // _FIXEDSIZEPOOL_HPP
