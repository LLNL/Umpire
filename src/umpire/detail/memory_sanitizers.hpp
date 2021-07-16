//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/config.hpp"

#if (defined(__clang__) || defined(__GNUC__)) && defined(UMPIRE_ENABLE_SANITIZERS)

#include <sanitizer/asan_interface.h>

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->get_platform() == umpire::Platform::host) {\
    ASAN_POISON_MEMORY_REGION((ptr), (size));\
  }

#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size) \
  if (allocator->get_platform() == umpire::Platform::host) {\
    ASAN_UNPOISON_MEMORY_REGION((ptr), (size));\
  }

#else

#define UMPIRE_POISON_MEMORY_REGION(allocator, ptr, size)
#define UMPIRE_UNPOISON_MEMORY_REGION(allocator, ptr, size)

#endif
