//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MpiSharedMemoryMemsetOperation_HPP
#define UMPIRE_MpiSharedMemoryMemsetOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Memset an allocation in MPI memory with provided function.  This is
 * an MPI collective call and this function will insure that guarded_fun is
 * only called by the foreman rank.
 */
class MpiSharedMemoryMemsetOperation : public MemoryOperation {
    public:
        void apply(
            void* src_ptr,
            util::AllocationRecord* allocation,
            std::size_t length,
            std::function<void (void*)> set_fun);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MpiSharedMemoryMemsetOperation_HPP
