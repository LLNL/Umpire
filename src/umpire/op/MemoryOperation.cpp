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
#include "umpire/op/MemoryOperation.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
MemoryOperation::transform(
    void* UMPIRE_UNUSED_ARG(src_ptr),
    void** UMPIRE_UNUSED_ARG(dst_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR("MemoryOperation::transform() is not implemented");
}

void
MemoryOperation::apply(
    void* UMPIRE_UNUSED_ARG(src_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    int UMPIRE_UNUSED_ARG(val),
    size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR("MemoryOperation::apply() is not implemented");
}



} // end of namespace op
} // end of namespace umpire
