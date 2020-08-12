//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTraits_HPP
#define UMPIRE_MemoryResourceTraits_HPP

#include <cstddef>

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_SYCL)
#include <CL/sycl.hpp>
#endif

namespace umpire {

struct MemoryResourceTraits {
  enum class optimized_for { any, latency, bandwidth, access };

  enum class vendor_type { UNKNOWN, AMD, IBM, INTEL, NVIDIA };

  enum class memory_type { UNKNOWN, DDR, GDDR, HBM, NVME };

  int id;

  // variables for only SYCL devices (i.e., Intel GPUs)
#if defined(UMPIRE_ENABLE_SYCL)
  cl::sycl::queue queue;
#endif

  bool unified = false;
  std::size_t size = 0;

  vendor_type vendor = vendor_type::UNKNOWN;
  memory_type kind = memory_type::UNKNOWN;
  optimized_for used_for = optimized_for::any;
};

} // end of namespace umpire

#endif // UMPIRE_MemoryResourceTraits_HPP
