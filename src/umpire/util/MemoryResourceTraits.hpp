//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTraits_HPP
#define UMPIRE_MemoryResourceTraits_HPP

#include <cstddef>
#include <string>

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_SYCL)
#include "umpire/util/sycl_compat.hpp"
#endif

namespace umpire {

struct MemoryResourceTraits {
  MemoryResourceTraits(){};

  enum class optimized_for { any, latency, bandwidth, access };

  enum class vendor_type { unknown, amd, ibm, intel, nvidia };

  enum class memory_type { unknown, ddr, gddr, hbm, nvme };

  enum class resource_type { unknown, host, device, device_const, pinned, um, file, shared };

  enum class shared_scope { unknown, node, socket };

  enum class granularity_type { unknown, fine_grained, coarse_grained };

  int id;

  // variables for only SYCL devices (i.e., Intel GPUs)
#if defined(UMPIRE_ENABLE_SYCL)
  sycl::queue* queue = nullptr;
#endif

  bool unified = false;
  bool ipc = false;

  std::size_t size = 0;

  vendor_type vendor = vendor_type::unknown;
  memory_type kind = memory_type::unknown;
  optimized_for used_for = optimized_for::any;
  resource_type resource = resource_type::unknown;
  shared_scope scope = shared_scope::unknown;
  granularity_type granularity = granularity_type::unknown;
  bool tracking{true};
};

inline std::string to_string(const MemoryResourceTraits::shared_scope& scope)
{
  switch (scope) {
    case MemoryResourceTraits::shared_scope::unknown:
      return "unknown";
    case MemoryResourceTraits::shared_scope::node:
      return "node";
    case MemoryResourceTraits::shared_scope::socket:
      return "socket";
  }
  return "unknown";
}

inline std::string to_string(const MemoryResourceTraits::optimized_for& value)
{
  switch (value) {
    case MemoryResourceTraits::optimized_for::any:
      return "any";
    case MemoryResourceTraits::optimized_for::latency:
      return "latency";
    case MemoryResourceTraits::optimized_for::bandwidth:
      return "bandwidth";
    case MemoryResourceTraits::optimized_for::access:
      return "access";
  }
  return "any";
}

inline std::string to_string(const MemoryResourceTraits::vendor_type& vendor)
{
  switch (vendor) {
    case MemoryResourceTraits::vendor_type::unknown:
      return "unknown";
    case MemoryResourceTraits::vendor_type::amd:
      return "amd";
    case MemoryResourceTraits::vendor_type::ibm:
      return "ibm";
    case MemoryResourceTraits::vendor_type::intel:
      return "intel";
    case MemoryResourceTraits::vendor_type::nvidia:
      return "nvidia";
  }
  return "unknown";
}

inline std::string to_string(const MemoryResourceTraits::memory_type& type)
{
  switch (type) {
    case MemoryResourceTraits::memory_type::unknown:
      return "unknown";
    case MemoryResourceTraits::memory_type::ddr:
      return "ddr";
    case MemoryResourceTraits::memory_type::gddr:
      return "gddr";
    case MemoryResourceTraits::memory_type::hbm:
      return "hbm";
    case MemoryResourceTraits::memory_type::nvme:
      return "nvme";
  }
  return "unknown";
}

inline std::string to_string(const MemoryResourceTraits::resource_type& resource)
{
  switch (resource) {
    case MemoryResourceTraits::resource_type::unknown:
      return "unknown";
    case MemoryResourceTraits::resource_type::host:
      return "host";
    case MemoryResourceTraits::resource_type::device:
      return "device";
    case MemoryResourceTraits::resource_type::device_const:
      return "device_const";
    case MemoryResourceTraits::resource_type::pinned:
      return "pinned";
    case MemoryResourceTraits::resource_type::um:
      return "um";
    case MemoryResourceTraits::resource_type::file:
      return "file";
    case MemoryResourceTraits::resource_type::shared:
      return "shared";
  }
  return "unknown";
}

inline std::string to_string(const MemoryResourceTraits::granularity_type& granularity)
{
  switch (granularity) {
    case MemoryResourceTraits::granularity_type::unknown:
      return "unknown";
    case MemoryResourceTraits::granularity_type::fine_grained:
      return "fine_grained";
    case MemoryResourceTraits::granularity_type::coarse_grained:
      return "coarse_grained";
  }
  return "unknown";
}

} // end of namespace umpire

#endif // UMPIRE_MemoryResourceTraits_HPP
