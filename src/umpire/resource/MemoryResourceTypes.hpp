//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

#include <cstddef>
#include <string>

#include "umpire/config.hpp"
#include "umpire/util/error.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#else
#include <regex>
#endif /* UMPIRE_ENABLE_CUDA */

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif /* UMPIRE_ENABLE_HIP */

namespace umpire {
namespace resource {

struct MemoryResourceTypeHash {
  template <typename T>
  std::size_t operator()(T t) const noexcept
  {
    return static_cast<std::size_t>(t);
  }
};

enum MemoryResourceType { Host, Device, Unified, Pinned, Constant, File, NoOp, Shared, Unknown };

inline std::string resource_to_string(MemoryResourceType type)
{
  switch (type) {
    case Host:
      return "HOST";
    case Device:
      return "DEVICE";
    case Unified:
      return "UM";
    case Pinned:
      return "PINNED";
    case Constant:
      return "DEVICE_CONST";
    case File:
      return "FILE";
    case NoOp:
      return "NO_OP";
    case Shared:
      return "SHARED";
    default:
      UMPIRE_ERROR(runtime_error, fmt::format("Unknown resource type: {}", static_cast<int>(type)));
  }

    //
    // The UMPIRE_ERROR macro above does not return.  It instead throws
    // an exception.  However, for some reason, nvcc throws a warning
    // "warning: missing return statement at end of non-void function"
    // even though the following line cannot be reached.  Adding this
    // fake return statement to work around the incorrect warning.
    //
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  return "Unknown";
#endif
}

inline MemoryResourceType string_to_resource(const std::string& resource)
{
  if (resource == "HOST")
    return MemoryResourceType::Host;
  else if (resource == "DEVICE")
    return MemoryResourceType::Device;
  else if (resource == "UM")
    return MemoryResourceType::Unified;
  else if (resource == "PINNED")
    return MemoryResourceType::Pinned;
  else if (resource == "DEVICE_CONST")
    return MemoryResourceType::Constant;
  else if (resource == "FILE")
    return MemoryResourceType::File;
  else if (resource == "NO_OP")
    return MemoryResourceType::NoOp;
  else if (resource == "SHARED")
    return MemoryResourceType::Shared;
  else {
    UMPIRE_ERROR(runtime_error, fmt::format("Unknown resource name \"{}\"", resource));
  }

  //
  // The UMPIRE_ERROR macro above does not return.  It instead throws
  // an exception.  However, for some reason, nvcc throws a warning
  // "warning: missing return statement at end of non-void function"
  // even though the following line cannot be reached.  Adding this
  // fake return statement to work around the incorrect warning.
  //
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  return MemoryResourceType::Unknown;
#endif
}

inline int resource_to_device_id(const std::string& resource)
{
  int device_id{0};

#if defined(UMPIRE_ENABLE_CUDA)
  if (resource.find("::") != std::string::npos) {
    device_id = std::stoi(resource.substr(resource.find("::") + 2));
  }
#else
  const std::regex id_regex{R"(.*::(\d+))", std::regex_constants::ECMAScript | std::regex_constants::optimize};
  std::smatch m;

  if (std::regex_match(resource, m, id_regex)) {
    device_id = std::stoi(m[1]);
  }
#endif
  else {
    // get the device bound to the current process

#if defined(UMPIRE_ENABLE_CUDA)
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDevice failed with error: {}", cudaGetErrorString(err)));
    }
#endif /* UMPIRE_ENABLE_CUDA */

#if defined(UMPIRE_ENABLE_HIP)
    hipError_t err = hipGetDevice(&device_id);
    if (err != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipGetDevice failed with error: {}", hipGetErrorString(err)));
    }
#endif /* UMPIRE_ENABLE_HIP */
  }

  return device_id;
}

} // end of namespace resource
} // end of namespace umpire

#endif
