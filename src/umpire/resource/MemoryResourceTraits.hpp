//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
namespace resource {

struct MemoryResourceTraits {

  enum class optimized_for {
    any,
    latency,
    bandwidth,
    access
  };

  enum class vendor_type {
    AMD,
    IBM,
    INTEL,
    NVIDIA
  };

  enum class memory_type {
    DDR,
    GDDR,
    HBM,
    NVME
  };

  bool unified;
  size_t size;

  vendor_type vendor;
  memory_type kind;
  optimized_for used_for;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceTraits_HPP
