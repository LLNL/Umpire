//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef _STDALLOCATOR_HPP
#define _STDALLOCATOR_HPP

#include <cstdlib>

struct StdAllocator {
  static inline void* allocate(std::size_t size)
  {
    return std::malloc(size);
  }
  static inline void deallocate(void* ptr)
  {
    std::free(ptr);
  }
};

#endif
