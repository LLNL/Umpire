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
#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
CudaMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  ::cudaMemset(src_ptr, value, length);
}

} // end of namespace op
} // end of namespace umpire
