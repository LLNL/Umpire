//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include <type_traits>

#include "camp/resource/platform.hpp"
#include "umpire/platform.hpp"

static_assert(std::is_empty<umpire::host_platform>::value, "platform tags must be empty");
static_assert(sizeof(umpire::host_platform) == 1, "empty tag size should be 1");
static_assert(umpire::platform_for<umpire::host_platform>::value == camp::resources::Platform::host,
              "host_platform mapping");
static_assert(umpire::platform_for<umpire::undefined_platform>::value == camp::resources::Platform::undefined,
              "undefined_platform mapping");

#if defined(UMPIRE_ENABLE_CUDA)
static_assert(umpire::platform_for<umpire::cuda_platform>::value == camp::resources::Platform::cuda,
              "cuda_platform mapping");
#endif

#if defined(UMPIRE_ENABLE_HIP)
static_assert(umpire::platform_for<umpire::hip_platform>::value == camp::resources::Platform::hip,
              "hip_platform mapping");
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
static_assert(umpire::platform_for<umpire::omp_target_platform>::value == camp::resources::Platform::omp_target,
              "omp_target_platform mapping");
#endif

TEST(platform_v2, Compiles)
{
  SUCCEED();
}

