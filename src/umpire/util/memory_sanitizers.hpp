//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#include "umpire/config.hpp"

#undef __UMPIRE_USE_MEMORY_SANITIZER__ // This may be defined below

#if defined(UMPIRE_HAS_ASAN)
//
// When a user compiles with "-fsanitize=memory", a couple of macros may be
// set depending upon compiler and version.
//
#include <sanitizer/asan_interface.h>

#if defined(__SANITIZE_ADDRESS__)
#undef __UMPIRE_USE_MEMORY_SANITIZER__
#define __UMPIRE_USE_MEMORY_SANITIZER__
#endif // defined(__SANITIZE_ADDRESS__)

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#undef __UMPIRE_USE_MEMORY_SANITIZER__
#define __UMPIRE_USE_MEMORY_SANITIZER__
#endif // __has_feature(address_sanitizer)
#endif // defined(__has_feature)

#endif // defined(UMPIRE_HAS_ASAN)

#if defined(__UMPIRE_USE_MEMORY_SANITIZER__)

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)   \
  if (allocator->getPlatform() == umpire::Platform::host) { \
    ASAN_POISON_MEMORY_REGION((ptr), (size));               \
  }

#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::host) { \
    ASAN_UNPOISON_MEMORY_REGION((ptr), (size));             \
  }

#else // !defined(__UMPIRE_USE_MEMORY_SANITIZER__)

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)
#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size)

#endif // defined(__UMPIRE_USE_MEMORY_SANITIZER__)

#endif // UMPIRE_memory_sanitizers_HPP
