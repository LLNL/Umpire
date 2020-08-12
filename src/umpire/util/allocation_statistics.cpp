//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/allocation_statistics.hpp"

#include <algorithm>

namespace umpire {
namespace util {

float relative_fragmentation(std::vector<util::AllocationRecord>& recs)
{
  auto r1 = recs.begin();
  auto r2 = recs.begin();
  ++r2;
  std::size_t largest_free_space = 0;
  std::size_t total_free_space = 0;
  while (r2 != recs.end()) {
    const std::size_t free_space =
        reinterpret_cast<char*>(r2->ptr) -
        (reinterpret_cast<char*>(r1->ptr) + r1->size);
    largest_free_space = std::max(largest_free_space, free_space);
    total_free_space += free_space;
  }

  return 1.0f - static_cast<float>(largest_free_space) /
                    (total_free_space + std::numeric_limits<float>::epsilon());
}

} // namespace util
} // namespace umpire
