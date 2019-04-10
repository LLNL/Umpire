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
#ifndef UMPIRE_allocation_statistics_HPP
#define UMPIRE_allocation_statistics_HPP

#include "umpire/util/AllocationRecord.hpp"

#include <vector>

namespace umpire {
namespace util {

/*!
 * \brief
 */
float relative_fragmentation(std::vector<util::AllocationRecord>& recs);

}
}

#endif // UMPIRE_allocation_statistics_HPP
