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
#include "umpire/op/CudaAdviseReadMostly.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace op {

void
CudaAdviseReadMostly::apply(
    void* src_ptr,
    util::AllocationRecord *src_allocation,
    int val,
    size_t length)
{
  // TODO: get correct device for allocation
  cudaMemAdvise(src_ptr, cudaMemAdviseSetReadMostly, 0);
}

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaAdviseReadMostly_HPP
