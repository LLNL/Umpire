//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipCopyFromOperation_HPP
#define UMPIRE_HipCopyFromOperation_HPP

#include "umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Copy operation to move data from a AMD GPU to CPU memory.
 */
class HipCopyFromOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::transform
   *
   * Uses hipMemcpy to move data when src_ptr is on a AMD GPU and dst_ptr
   * is on the CPU.
   *
   * @copydetails MemoryOperation::transform
   */
  void transform(void* src_ptr, void** dst_ptr,
                 util::AllocationRecord* src_allocation,
                 util::AllocationRecord* dst_allocation, std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_HipCopyFromOperation_HPP
