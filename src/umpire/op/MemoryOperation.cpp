//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/MemoryOperation.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void MemoryOperation::transform(
    void* UMPIRE_UNUSED_ARG(src_ptr), void** UMPIRE_UNUSED_ARG(dst_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR("MemoryOperation::transform() is not implemented");
}

camp::resources::Event MemoryOperation::transform_async(
    void* UMPIRE_UNUSED_ARG(src_ptr), void** UMPIRE_UNUSED_ARG(dst_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t UMPIRE_UNUSED_ARG(length),
    camp::resources::Resource& UMPIRE_UNUSED_ARG(ctx))
{
  UMPIRE_ERROR("MemoryOperation::transform() is not implemented");
}

void MemoryOperation::apply(
    void* UMPIRE_UNUSED_ARG(src_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    int UMPIRE_UNUSED_ARG(val), std::size_t UMPIRE_UNUSED_ARG(length))
{
  UMPIRE_ERROR("MemoryOperation::apply() is not implemented");
}

camp::resources::Event MemoryOperation::apply_async(
    void* UMPIRE_UNUSED_ARG(src_ptr),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    int UMPIRE_UNUSED_ARG(val), std::size_t UMPIRE_UNUSED_ARG(length),
    camp::resources::Resource& UMPIRE_UNUSED_ARG(ctx))
{
  UMPIRE_ERROR("MemoryOperation::apply() is not implemented");
}

} // end of namespace op
} // end of namespace umpire
