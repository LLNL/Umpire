//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#undef __UMPIRE_USE_MEMORY_SANITIZER__ // This may be defined below

//
// When a user compiles with "-fsanitize=memory", a couple of macros may be
// set depending upon compiler and version.  The two compilers that most
// broadly understand this option are clang and GNUC.  Exceptions to this are:
//
// GNUC 4 does not know about asan_interface.h.  Nor does the XL compiler,
// even when it is using the clang front-end.
//
// The SYCL compiler (at least the one as of this writing) also does not
// seem to know where <sanitizer/asan_interface> is so we exclude its inclusion.
//
#if (defined(__clang__) && !defined(__ibmxl__)) || \
    (defined(__GNUC__) && __GNUC__ > 4)

#if !defined(__SYCL_COMPILER_VERSION)

#include <sanitizer/asan_interface.h>

#if defined(__SANITIZE_ADDRESS__)

#undef __UMPIRE_USE_MEMORY_SANITIZER__
#define __UMPIRE_USE_MEMORY_SANITIZER__

#endif // #if defined(__SANITIZE_ADDRESS__)

#if defined(__has_feature)
#if __has_feature(address_sanitizer)

#undef __UMPIRE_USE_MEMORY_SANITIZER__
#define __UMPIRE_USE_MEMORY_SANITIZER__

#endif // #if __has_feature(address_sanitizer)
#endif // #if defined(__has_feature)

#endif // #if !defined(__SYCL_COMPILER_VERSION)

#endif // #if (defined(__clang__) && !defined(__ibmxl__))  || (defined(__GNUC__)
       // && __GNUC__ > 4)

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

#endif // #if defined(__UMPIRE_USE_MEMORY_SANITIZER__)

#endif // #ifndef UMPIRE_memory_sanitizers_HPP
