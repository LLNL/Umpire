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
#ifndef UMPIRE_Numa_HPP
#define UMPIRE_Numa_HPP

#include <cstddef>
#include <vector>

namespace umpire {
namespace numa {

// Used in PosixMemalignAllocator but static is defined here
extern long s_cpu_page_size;

// Return the preferred numa node
int preferred_node();

} // end namespace numa
} // end namespace umpire


#endif // UMPIRE_Numa_HPP
