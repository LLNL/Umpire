//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/MpiSharedMemoryMemsetOperation.hpp"
#include "umpire/resource/MpiSharedMemoryResource.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/wrap_allocator.hpp"

namespace umpire {
namespace op {

void
MpiSharedMemoryMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord* allocation,
    std::size_t UMPIRE_UNUSED_ARG(length),
    std::function<void (void*)> set_fun)
{
    auto mpi_allocator {
        umpire::util::unwrap_allocation_strategy<resource::MpiSharedMemoryResource>(allocation->strategy)
    };


    mpi_allocator->fence(src_ptr);

    if ( mpi_allocator->isForeman() ) {
        set_fun(src_ptr);
    }

    mpi_allocator->fence(src_ptr);

    UMPIRE_RECORD_STATISTIC(
        "MpiSharedMemoryMemsetOperation",
        "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
        "size", length,
        "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
