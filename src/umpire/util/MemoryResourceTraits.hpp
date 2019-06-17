//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTraits_HPP
#define UMPIRE_MemoryResourceTraits_HPP

#include <cstddef>

namespace umpire {

struct MemoryResourceTraits {

  enum class optimized_for {
    any,
    latency,
    bandwidth,
    access
  };

  enum class vendor_type {
    UNKNOWN,
    AMD,
    IBM,
    INTEL,
    NVIDIA
  };

  enum class memory_type {
    UNKNOWN,
    DDR,
    GDDR,
    HBM,
    NVME
  };

  bool unified = false;
  std::size_t size = 0;

  vendor_type vendor = vendor_type::UNKNOWN;
  memory_type kind = memory_type::UNKNOWN;
  optimized_for used_for = optimized_for::any;
};

} // end of namespace umpire

#endif // UMPIRE_MemoryResourceTraits_HPP
