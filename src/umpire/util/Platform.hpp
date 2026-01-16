//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Platform_HPP
#define UMPIRE_Platform_HPP

#include <string>

#include "camp/resource/platform.hpp"

namespace umpire {

using Platform = camp::resources::Platform;

inline std::string platform_to_string(Platform type)
{
  switch (type) {
    case Platform::undefined:
      return "Undefined";
    case Platform::host:
      return "Host";
    case Platform::cuda:
      return "Cuda";
    case Platform::omp_target:
      return "OmpTarget";
    case Platform::hip:
      return "Hip";
    case Platform::sycl:
      return "Sycl";
    default:
      return "Unknown";
  }
}

} // end of namespace umpire

#endif
