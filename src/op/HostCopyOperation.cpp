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
#include "umpire/op/HostCopyOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Statistic.hpp"

namespace umpire {
namespace op {

void HostCopyOperation::transform(
    void* src_ptr,
    void* dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  util::Statistic::OperationStatistic stat = {src_ptr, dst_ptr, length, "transform"};
  util::StatisticsDatabase::getDatabase()->getStatistic("HostCopyOperation", util::Statistic::OP_STAT)->recordOperationStatistic(stat);

  std::memcpy(
      dst_ptr,
      src_ptr,
      length);
}

} // end of namespace op
} // end of namespace umpire
