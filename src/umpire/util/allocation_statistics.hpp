//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_allocation_statistics_HPP
#define UMPIRE_allocation_statistics_HPP

#include "umpire/util/AllocationRecord.hpp"

#include <vector>

namespace umpire {
namespace util {

/*!
 * \brief Compute the relative fragmentation of a set of allocation records.
 *
 * Fragmentation = 1 - (largest free block) / (total free space)
 */
float relative_fragmentation(std::vector<util::AllocationRecord>& recs);

}
}

#endif // UMPIRE_allocation_statistics_HPP
