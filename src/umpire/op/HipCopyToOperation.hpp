//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipCopyToOperation_HPP
#define UMPIRE_HipCopyToOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data from CPU to AMD GPU memory.
 */
class HipCopyToOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses hipMemcpy to move data when src_ptr is on the CPU and dst_ptr
   * is on an AMD GPU.
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 umpire::util::AllocationRecord* src_allocation,
                 umpire::util::AllocationRecord* dst_allocation,
                 std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_HipCopyToOperation_HPP
