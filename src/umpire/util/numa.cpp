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
#include "umpire/util/numa.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>
#include <numaif.h>
#include <unistd.h>

namespace umpire {

long s_cpu_page_size = sysconf(_SC_PAGESIZE);

namespace numa {

int preferred_node() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_preferred();
}

void move_to_node(void *ptr, size_t bytes, int node) {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  struct bitmask *mask = numa_bitmask_alloc(numa_max_node() + 1);
  numa_bitmask_clearall(mask);
  numa_bitmask_setbit(mask, node);

  if (mbind(ptr, bytes, MPOL_BIND, mask->maskp, mask->size + 1, MPOL_MF_STRICT) != 0) {
    UMPIRE_ERROR("numa::move_to_node error: mbind( ptr = " << ptr <<
                 ", bytes = " << bytes <<
                 ", node = " << node << " ) failed");
  }

  numa_bitmask_free(mask);
}


} // end namespace numa
} // end namespace umpire
