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
#include "umpire/util/Numa.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>
#include <unistd.h>

namespace umpire {

long s_cpu_page_size = sysconf(_SC_PAGESIZE);

namespace numa {

int preferred_node() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_preferred();
}

} // end namespace numa
} // end namespace umpire
