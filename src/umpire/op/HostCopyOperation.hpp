//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HostCopyOperation_HPP
#define UMPIRE_HostCopyOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy memory between two allocations in CPU memory.
 */
class HostCopyOperation : public MemoryOperation {
 public:
  /*
   * \copybrief MemoryOperation::transform
   *
   * Perform a memcpy to move length bytes of data from src_ptr to dst_ptr
   *
   * \copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 umpire::util::AllocationRecord* src_allocation,
                 umpire::util::AllocationRecord* dst_allocation,
                 std::size_t length);
};

} // namespace op
} // end of namespace umpire

#endif // UMPIRE_HostCopyOperation_HPP
