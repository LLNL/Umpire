//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_sanitizers_HPP
#define UMPIRE_memory_sanitizers_HPP

#include "umpire/config.hpp"

#if defined(__clang__) && defined(UMPIRE_ENABLE_SANITIZERS)

#include <sanitizer/asan_interface.h>

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::cpu) {\
    ASAN_POISON_MEMORY_REGION((ptr), (size));\
  }

#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->getPlatform() == umpire::Platform::cpu) {\
    ASAN_UNPOISON_MEMORY_REGION((ptr), (size));\
  }

#else

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)
#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size)

#endif

#endif // UMPIRE_memory_sanitizers_HPP