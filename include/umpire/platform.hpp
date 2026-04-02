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

//! Re-export the host platform tag in the top-level namespace.
using resource::host_platform;
//! Re-export the compile-time platform mapping helper.
using resource::platform_for;
//! Re-export the undefined platform tag in the top-level namespace.
using resource::undefined_platform;

#if defined(UMPIRE_ENABLE_CUDA)
//! Re-export the CUDA platform tag in the top-level namespace.
using resource::cuda_platform;
#endif
#if defined(UMPIRE_ENABLE_HIP)
//! Re-export the HIP platform tag in the top-level namespace.
using resource::hip_platform;
#endif
#if defined(UMPIRE_ENABLE_SYCL)
//! Re-export the SYCL platform tag in the top-level namespace.
using resource::sycl_platform;
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
//! Re-export the OpenMP target platform tag in the top-level namespace.
using resource::omp_target_platform;
#endif

} // namespace umpire

#endif // UMPIRE_platform_HPP
