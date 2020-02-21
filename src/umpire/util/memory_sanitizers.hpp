//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#if defined(__clang__)
#include <sanitizer/asan_interface.h>

#define POISON_MEMORY_IF_CPU(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::cpu) {\
    ASAN_POISON_MEMORY_REGION((ptr), (size));\
  }

#define UNPOISON_MEMORY_IF_CPU(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::cpu) {\
    ASAN_UNPOISON_MEMORY_REGION((ptr), (size));\
  }

#else

#define POISON_MEMORY_IF_CPU(allocator, ptr, size)
#define UNPOISON_MEMORY_IF_CPU(allocator, ptr, size)

#endif

#endif // UMPIRE_memory_sanitizers_HPP