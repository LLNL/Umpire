//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#include "umpire/config.hpp"

#if defined(__clang__) || defined(__GNUC__)

#include <sanitizer/asan_interface.h>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::host) {\
    ASAN_POISON_MEMORY_REGION((ptr), (size));\
  }

#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::host) {\
    ASAN_UNPOISON_MEMORY_REGION((ptr), (size));\
  }
#else
#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)
#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size)
#endif // __has_feature(address_sanitizer)
#else
#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)
#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size)
#endif // defined(__has_feature)
#endif  // defined(__clang__) || defined(__GNUC__)

#endif // UMPIRE_memory_sanitizers_HPP
