//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_platform_HPP
#define UMPIRE_platform_HPP

#include "umpire/resource/platform.hpp"

namespace umpire {

using resource::host_platform;
using resource::platform_for;
using resource::undefined_platform;

#if defined(UMPIRE_ENABLE_CUDA)
using resource::cuda_platform;
#endif
#if defined(UMPIRE_ENABLE_HIP)
using resource::hip_platform;
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
using resource::omp_target_platform;
#endif

} // namespace umpire

#endif // UMPIRE_platform_HPP
