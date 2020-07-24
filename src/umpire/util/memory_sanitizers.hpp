//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#include "umpire/config.hpp"

#if (defined(__clang__) && !defined(__ibmxl__))  || (defined(__GNUC__) && __GNUC__ > 4)
#include <sanitizer/asan_interface.h>
#endif

#undef UMPIRE_USE_MEMORY_SANITIZER

#if defined(__SANITIZE_ADDRESS__)
#undef UMPIRE_USE_MEMORY_SANITIZER
#define UMPIRE_USE_MEMORY_SANITIZER
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#undef UMPIRE_USE_MEMORY_SANITIZER
#define UMPIRE_USE_MEMORY_SANITIZER
#endif // defined(__has_feature)
#endif  // defined(__clang__) || defined(__GNUC__)

#if defined(UMPIRE_USE_MEMORY_SANITIZER)
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

#endif // UMPIRE_memory_sanitizers_HPP
