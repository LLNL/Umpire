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

#include "umpire/util/allocation_statistics.hpp"

#include <algorithm>

namespace umpire {
namespace util {

float relative_fragmentation(std::vector<util::AllocationRecord>& recs)
{
  auto r1 = recs.begin();
  auto r2 = recs.begin(); ++r2;
  std::size_t largest_free_space = 0;
  std::size_t total_free_space = 0;
  while (r2 != recs.end()) {
    const std::size_t free_space =
      reinterpret_cast<char*>(r2->m_ptr) - (reinterpret_cast<char*>(r1->m_ptr) + r1->m_size);
    largest_free_space = std::max(largest_free_space, free_space);
    total_free_space += free_space;
  }

  return 1.0 - static_cast<float>(largest_free_space) / total_free_space;
}

}
}
